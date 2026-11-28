"""Persist observation metadata in SQLite and source/crop PNGs on disk."""

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from parking_monitoring.schemas import Detection, ImageResult, OCRResult, PlateResult


@dataclass
class ObservationSummary:
    """History entry with its observation ID, source ID and stored UTC timestamp."""

    observation_id: int
    source_id: str
    created_at: str


@dataclass
class ImageFiles:
    """Saved source and crop paths, with crops in result.plates order."""

    source_path: Path
    crop_paths: list[Path]


class SQLiteStorage:
    """Coordinate local observation persistence in SQLite and image files.

    save stores metadata only; save_with_images also persists source/crop PNGs.
    Each explicit save creates a new observation. Connections close after use.
    """

    def __init__(self, database_path: Path) -> None:
        """Prepare the database and tables without replacing existing data.

        Raises:
            ValueError: The path is not a Path or refers to a directory.
            OSError: The parent directory cannot be prepared.
            sqlite3.Error: The database cannot be opened or initialized.
        """
        # 1. Validate the file path and prepare the database table.
        if not isinstance(database_path, Path):
            raise ValueError("database_path must be a pathlib.Path")  # noqa: TRY004
        self.database_path = database_path.expanduser().resolve()
        if self.database_path.is_dir():
            raise ValueError("database_path must refer to a file, not a directory")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS observation_files (
                    observation_id INTEGER PRIMARY KEY REFERENCES observations(id),
                    source_path TEXT NOT NULL,
                    crop_paths_json TEXT NOT NULL
                )
            """)

    def save(self, result: ImageResult) -> int:
        """Commit a complete result and return its newly generated observation ID.

        Raises:
            ValueError: The input is not a usable, JSON-serializable ImageResult.
            sqlite3.Error: Insert or commit fails; no success ID is returned.
        """
        # 2. Share insertion; commit/rollback before closing the connection.
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            observation_id = self._insert_observation(connection, result)
        # The transaction context commits before execution reaches this return.
        return observation_id

    def get(self, observation_id: int) -> ImageResult | None:
        """Reconstruct a saved ImageResult, or return None only for an absent ID.

        Raises:
            ValueError: ID is not a positive SQLite integer, or saved data is malformed.
            sqlite3.Error: The database read fails.
        """
        # 4. Validate the ID and find the observation using a parameterized query.
        if type(observation_id) is not int or not 0 < observation_id <= 2**63 - 1:
            raise ValueError(
                "observation_id must be a positive SQLite integer, excluding bool"
            )
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT result_json FROM observations WHERE id = ?",
                (observation_id,),
            ).fetchone()
        if row is None:
            return None

        # 5. Reconstruct nested dataclasses and tuples; distinguish corrupt data from absence.
        try:
            data = json.loads(row[0])
            plates = []
            for item in data["plates"]:
                detection = Detection(
                    bbox_xyxy=tuple(item["detection"]["bbox_xyxy"]),
                    score=item["detection"]["score"],
                )
                plates.append(
                    PlateResult(
                        detection=detection,
                        crop_bbox_xyxy=tuple(item["crop_bbox_xyxy"]),
                        ocr=OCRResult(**item["ocr"]),
                    )
                )
            return ImageResult(
                source_id=data["source_id"],
                image_size_wh=tuple(data["image_size_wh"]),
                plates=plates,
                timings_ms=data["timings_ms"],
                model_info=data["model_info"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot decode observation {observation_id}: {exc}"
            ) from exc

    def _insert_observation(
        self,
        connection: sqlite3.Connection,
        result: ImageResult,
    ) -> int:
        """Validate and insert metadata; return its ID without committing."""
        # 3. Serialize before INSERT; leave the transaction to the public caller.
        if not isinstance(result, ImageResult):
            raise ValueError("result must be an ImageResult")  # noqa: TRY004
        if not isinstance(result.source_id, str) or not result.source_id.strip():
            raise ValueError("result.source_id must be a nonblank string")
        try:
            result_json = json.dumps(asdict(result), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot serialize ImageResult: {exc}") from exc
        cursor = connection.execute(
            """
            INSERT INTO observations (source_id, created_at, result_json)
            VALUES (?, ?, ?)
            """,
            (result.source_id, datetime.now(UTC).isoformat(), result_json),
        )
        observation_id = cursor.lastrowid
        if observation_id is None:
            raise sqlite3.DatabaseError("Insert did not return an observation ID")
        return observation_id

    def save_with_images(
        self,
        image: Image.Image,
        result: ImageResult,
        *,
        sources_dir: Path,
        crops_dir: Path,
    ) -> int:
        """Save source/crop PNGs and metadata; return the committed ID.

        Require image.size to match result.image_size_wh. Use sources_dir and
        crops_dir as roots for fresh UUID folders; persist absolute paths.
        Save decoded pixels losslessly in PNG-compatible modes, without changing
        caller pixels. Crop paths follow plate order, including OCR errors.
        Zero plates saves the source and an empty crop list.

        Raises:
            ValueError: Image/result, PNG mode, directories or bounds are invalid.
            OSError: Image/file access or cleanup fails. Cleanup errors include
                residual paths and retain the original error as their cause.
            sqlite3.Error: Insert/commit fails, after attempt-owned cleanup.

        Ordinary failures roll back both rows and clean attempt-owned folders.
        Abrupt process termination can leave orphan files; filesystem writes are
        not part of SQLite's transaction.
        """
        # 4. Validate dimensions and exact crop bounds before creating files.
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("image must be a nonempty Pillow image")
        if not isinstance(result, ImageResult):
            raise ValueError("result must be an ImageResult")  # noqa: TRY004
        if (
            not isinstance(result.image_size_wh, tuple)
            or len(result.image_size_wh) != 2
            or any(type(value) is not int for value in result.image_size_wh)
            or image.size != result.image_size_wh
        ):
            raise ValueError("image.size must match result.image_size_wh")
        if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I;16"}:
            raise ValueError(
                f"Image mode {image.mode!r} cannot be saved losslessly as PNG"
            )
        if not isinstance(result.plates, list):
            raise ValueError("result.plates must be a list")  # noqa: TRY004
        for index, plate in enumerate(result.plates):
            if not isinstance(plate, PlateResult):
                raise ValueError(f"Plate {index} must be a PlateResult")  # noqa: TRY004
            bounds = plate.crop_bbox_xyxy
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 4
                or any(type(value) is not int for value in bounds)
                or not (0 <= bounds[0] < bounds[2] <= image.width)
                or not (0 <= bounds[1] < bounds[3] <= image.height)
            ):
                raise ValueError(
                    f"Plate {index} has invalid crop_bbox_xyxy: {bounds!r}"
                )
        if not isinstance(sources_dir, Path) or not isinstance(crops_dir, Path):
            raise ValueError("sources_dir and crops_dir must be pathlib.Path values")  # noqa: TRY004
        sources_root = sources_dir.expanduser().resolve()
        crops_root = crops_dir.expanduser().resolve()
        image.load()

        # 5. Allocate unique paths; never overwrite or clean an existing folder.
        folder_name = uuid4().hex
        source_folder = sources_root / folder_name
        crop_folder = crops_root / folder_name
        source_path = source_folder / "source.png"
        crop_paths = [
            crop_folder / f"plate_{index:03d}.png"
            for index in range(len(result.plates))
        ]
        owned_folders: list[Path] = []
        try:
            with closing(sqlite3.connect(self.database_path)) as connection, connection:
                connection.execute("PRAGMA foreign_keys = ON")
                observation_id = self._insert_observation(connection, result)
                for folder in (source_folder, *([crop_folder] if crop_paths else [])):
                    if folder not in owned_folders:
                        folder.parent.mkdir(parents=True, exist_ok=True)
                        folder.mkdir()
                        owned_folders.append(folder)

                # 6. Write independent original-resolution source/crop PNGs.
                for index, path in enumerate([source_path, *crop_paths]):
                    pixels = (
                        image.copy()
                        if index == 0
                        else image.crop(result.plates[index - 1].crop_bbox_xyxy)
                    )
                    with pixels:
                        try:
                            pixels.save(path, format="PNG")
                        except (OSError, ValueError) as exc:
                            raise OSError(
                                f"Cannot save image file {path}: {exc}"
                            ) from exc

                # 7. Commit metadata and file references together after all writes.
                connection.execute(
                    """
                    INSERT INTO observation_files
                        (observation_id, source_path, crop_paths_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        observation_id,
                        str(source_path),
                        json.dumps([str(p) for p in crop_paths]),
                    ),
                )
            return observation_id
        except BaseException as exc:
            # 8. Rollback/connection close happened above; clean only our folders.
            cleanup_errors = []
            for folder in reversed(owned_folders):
                try:
                    shutil.rmtree(folder)
                except OSError as cleanup_exc:
                    cleanup_errors.append(f"{folder}: {cleanup_exc}")
            if cleanup_errors:
                raise OSError(
                    f"Save failed ({exc}); cleanup failed; inspect remaining paths: "
                    + "; ".join(cleanup_errors)
                ) from exc
            raise

    def get_image_files(self, observation_id: int) -> ImageFiles | None:
        """Return source/crop paths for a positive observation ID.

        Return None for an absent ID or metadata-only observation.
        Raise ValueError for invalid IDs or malformed references.
        Raise OSError for registered files that are missing or inaccessible.
        """
        # 9. Retrieve registered paths; distinguish absence, corruption and file errors.
        if type(observation_id) is not int or not 0 < observation_id <= 2**63 - 1:
            raise ValueError(
                "observation_id must be a positive SQLite integer, excluding bool"
            )
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT f.source_path, f.crop_paths_json, o.result_json
                FROM observation_files AS f
                JOIN observations AS o ON o.id = f.observation_id
                WHERE o.id = ?
                """,
                (observation_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            crops = json.loads(row[1])
            plates = json.loads(row[2])["plates"]
            if (
                not isinstance(crops, list)
                or not isinstance(plates, list)
                or len(crops) != len(plates)
            ):
                raise ValueError("Crop paths must match the stored plate count")
            values = [row[0], *crops]
            if any(
                not isinstance(value, str)
                or not value.strip()
                or "\x00" in value
                or not Path(value).is_absolute()
                for value in values
            ):
                raise ValueError("File references must be nonblank absolute paths")
            paths = [Path(value) for value in values]
            if len(set(paths)) != len(paths):
                raise ValueError("File references must be distinct")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid image references for observation {observation_id}: {exc}"
            ) from exc
        for path in paths:
            # Opening verifies accessibility and rejects directory references.
            with path.open("rb"):
                pass
        return ImageFiles(source_path=paths[0], crop_paths=paths[1:])

    def list_recent(self, limit: int = 20) -> list[ObservationSummary]:
        """Return newest observations first, ordered by descending ID.

        Return an empty list when no observations exist. Include metadata-only
        observations; do not open image files or decode result payloads.

        Raises:
            ValueError: limit is not an integer from 1 to 100, excluding bool.
            sqlite3.Error: The database read fails.
        """
        # 10. Bound the history query and return lightweight metadata only.
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100, excluding bool")
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT id, source_id, created_at FROM observations
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [ObservationSummary(*row) for row in rows]
