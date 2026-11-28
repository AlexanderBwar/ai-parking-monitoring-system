"""Focused SQLite roundtrip, input-boundary and transaction-failure checks."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from PIL import Image

import parking_monitoring.storage as module
from parking_monitoring.schemas import Detection, ImageResult, OCRResult, PlateResult


@pytest.fixture
def result():
    return ImageResult(
        source_id="image'; DROP TABLE observations; --.jpg",
        image_size_wh=(30, 20),
        plates=[
            PlateResult(
                Detection((1.2, 2.4, 9.1, 8.2), 0.9),
                (1, 2, 10, 9),
                OCRResult("YGN A-1234", "YGNA-1234", 0.95, "read"),
            ),
            PlateResult(
                Detection((10, 2, 18, 10), 0.8),
                (10, 2, 18, 10),
                OCRResult(None, None, None, "error", "OCR backend failed"),
            ),
            PlateResult(
                Detection((20, 2, 28, 10), 0.7),
                (20, 2, 28, 10),
                OCRResult("MDY B-5678", "MDYB-5678", 0.4, "uncertain"),
            ),
            PlateResult(
                Detection((2, 10, 10, 18), 0.65),
                (2, 10, 10, 18),
                OCRResult(None, None, None, "unreadable"),
            ),
        ],
        timings_ms={"detection": 10, "cropping": 2, "ocr": 30, "total": 43},
        model_info={"detector_checkpoint": "local.pth", "ocr_model": "PP-OCRv4/en"},
    )


@pytest.mark.parametrize("empty", [False, True])
def test_complete_result_survives_reopening(tmp_path, result, empty):
    if empty:
        result = replace(result, plates=[])
    path = tmp_path / "nested" / "observations.sqlite3"
    storage = module.SQLiteStorage(path)
    observation_id = storage.save(result)
    assert type(observation_id) is int and observation_id > 0

    reopened = module.SQLiteStorage(path)
    restored = reopened.get(observation_id)
    assert restored == result
    assert restored is not result
    assert isinstance(restored.image_size_wh, tuple)
    if not empty:
        assert isinstance(restored.plates[0], PlateResult)
        assert isinstance(restored.plates[0].detection, Detection)
        assert isinstance(restored.plates[0].detection.bbox_xyxy, tuple)
        assert isinstance(restored.plates[0].crop_bbox_xyxy, tuple)
        assert isinstance(restored.plates[0].ocr, OCRResult)
        assert restored.plates[1].ocr.status == "error"
    assert reopened.get(observation_id + 1000) is None

    with closing(sqlite3.connect(path)) as connection:
        source_id, created_at, payload = connection.execute(
            "SELECT source_id, created_at, result_json FROM observations WHERE id = ?",
            (observation_id,),
        ).fetchone()
    assert source_id == result.source_id
    assert datetime.fromisoformat(created_at).utcoffset().total_seconds() == 0
    assert json.loads(payload)["source_id"] == result.source_id


def test_each_explicit_save_gets_its_own_id(tmp_path, result):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    first = storage.save(result)
    second = storage.save(result)
    assert first != second
    assert storage.get(first) == storage.get(second) == result


def test_recent_history_is_bounded_and_survives_reopening(tmp_path, result):
    path = tmp_path / "observations.sqlite3"
    storage = module.SQLiteStorage(path)
    assert storage.list_recent() == []
    ids = [storage.save(result) for _ in range(22)]

    reopened = module.SQLiteStorage(path)
    recent = reopened.list_recent()
    assert [item.observation_id for item in recent] == list(reversed(ids))[:20]
    assert all(item.source_id == result.source_id for item in recent)
    assert all(
        datetime.fromisoformat(item.created_at).utcoffset().total_seconds() == 0
        for item in recent
    )
    assert len(reopened.list_recent(100)) == 22
    assert reopened.list_recent(1) == recent[:1]
    assert reopened.get(recent[0].observation_id) == result


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.0, "20", None])
def test_recent_history_rejects_invalid_limits(tmp_path, limit):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    with pytest.raises(ValueError, match="limit"):
        storage.list_recent(limit)


def test_recent_history_propagates_database_errors(tmp_path):
    path = tmp_path / "observations.sqlite3"
    storage = module.SQLiteStorage(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP TABLE observations")
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        storage.list_recent()


@pytest.mark.parametrize("kind", ["none", "string", "directory"])
def test_invalid_database_path(tmp_path, kind):
    path = {"none": None, "string": "observations.sqlite3", "directory": tmp_path}[kind]
    with pytest.raises(ValueError, match="database_path"):
        module.SQLiteStorage(path)


def test_invalid_existing_file_is_not_replaced(tmp_path):
    path = tmp_path / "existing.sqlite3"
    original = b"This is not a SQLite database."
    path.write_bytes(original)
    with pytest.raises(sqlite3.DatabaseError):
        module.SQLiteStorage(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "kind", ["none", "blank-source", "wrong-source", "nan", "object"]
)
def test_invalid_save_does_not_insert(tmp_path, result, kind):
    invalid = {
        "none": None,
        "blank-source": replace(result, source_id="  "),
        "wrong-source": replace(result, source_id=123),
        "nan": replace(result, timings_ms={"total": float("nan")}),
        "object": replace(result, model_info={"bad": object()}),
    }[kind]
    path = tmp_path / "observations.sqlite3"
    storage = module.SQLiteStorage(path)
    with pytest.raises(ValueError):
        storage.save(invalid)
    with closing(sqlite3.connect(path)) as connection:
        assert (
            connection.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
        )


@pytest.mark.parametrize("observation_id", [None, True, 0, -1, "1", 2**63])
def test_invalid_lookup_id(tmp_path, observation_id):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    with pytest.raises(ValueError, match="observation_id"):
        storage.get(observation_id)


@pytest.mark.parametrize("payload", ["not JSON", "{}", "[]"])
def test_corrupt_record_is_an_error_not_a_missing_id(tmp_path, payload):
    path = tmp_path / "observations.sqlite3"
    storage = module.SQLiteStorage(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "INSERT INTO observations (source_id, created_at, result_json) VALUES (?, ?, ?)",
            ("broken.jpg", "2026-11-14T00:00:00+00:00", payload),
        )
    with pytest.raises(ValueError, match="Cannot decode observation 1"):
        storage.get(1)


def test_database_read_errors_are_not_reported_as_missing_ids(tmp_path):
    path = tmp_path / "observations.sqlite3"
    storage = module.SQLiteStorage(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP TABLE observations")
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        storage.get(1)


def test_missing_insert_id_raises_before_success(tmp_path, result, monkeypatch):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.return_value.lastrowid = None
    monkeypatch.setattr(module.sqlite3, "connect", lambda _: connection)
    with pytest.raises(sqlite3.DatabaseError, match="observation ID"):
        storage.save(result)
    assert connection.__exit__.call_args.args[0] is sqlite3.DatabaseError
    connection.close.assert_called_once()


def test_failed_insert_rolls_back_and_connection_is_released(tmp_path, result):
    path = tmp_path / "observations.sqlite3"
    storage = module.SQLiteStorage(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("""
            CREATE TRIGGER reject_save BEFORE INSERT ON observations
            BEGIN SELECT RAISE(ABORT, 'forced insert failure'); END
        """)
    with pytest.raises(sqlite3.IntegrityError, match="forced insert failure"):
        storage.save(result)
    with closing(sqlite3.connect(path)) as connection, connection:
        assert (
            connection.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
        )
        connection.execute("DROP TRIGGER reject_save")
    assert storage.get(storage.save(result)) == result


def test_failed_commit_rolls_back_without_returning_an_id(
    tmp_path, result, monkeypatch
):
    # A real deferred FK fails at commit after INSERT itself succeeds.
    path = tmp_path / "observations.sqlite3"
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    monkeypatch.setattr(module.sqlite3, "connect", connect)
    storage = module.SQLiteStorage(path)
    with closing(connect(path)) as connection, connection:
        connection.executescript("""
            CREATE TABLE parent (id INTEGER PRIMARY KEY);
            CREATE TABLE pending (
                parent_id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED
            );
            CREATE TRIGGER defer_failure AFTER INSERT ON observations
            BEGIN INSERT INTO pending VALUES (999); END;
        """)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        storage.save(result)
    with closing(connect(path)) as connection, connection:
        assert (
            connection.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
        )
        assert connection.execute("SELECT count(*) FROM pending").fetchone()[0] == 0
        connection.execute("DROP TRIGGER defer_failure")
    assert storage.get(storage.save(result)) == result


@pytest.fixture
def image():
    with Image.frombytes(
        "RGB", (30, 20), bytes(range(256)) * 7 + bytes(range(8))
    ) as pixels:
        yield pixels


@pytest.mark.parametrize("empty", [False, True])
def test_images_roundtrip_preserves_pixels_order_and_errors(
    tmp_path, result, image, empty
):
    if empty:
        result = replace(result, plates=[])
    original = image.tobytes()
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    observation_id = storage.save_with_images(
        image, result, sources_dir=tmp_path / "sources", crops_dir=tmp_path / "crops"
    )
    reopened = module.SQLiteStorage(storage.database_path)
    assert reopened.get(observation_id) == result
    files = reopened.get_image_files(observation_id)
    assert files.source_path.is_absolute()
    with Image.open(files.source_path) as source:
        assert source.size == image.size and source.tobytes() == original
    assert len(files.crop_paths) == len(result.plates)
    for plate, path in zip(result.plates, files.crop_paths, strict=True):
        assert path.is_absolute()
        with Image.open(path) as crop, image.crop(plate.crop_bbox_xyxy) as expected:
            assert crop.size == expected.size and crop.tobytes() == expected.tobytes()
    assert image.tobytes() == original
    if empty:
        assert not (tmp_path / "crops").exists()
    assert reopened.get_image_files(observation_id + 1000) is None
    assert reopened.get_image_files(reopened.save(result)) is None


def test_same_root_repeated_saves_never_overwrite(tmp_path, result, image):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    first = storage.save_with_images(
        image, result, sources_dir=tmp_path, crops_dir=tmp_path
    )
    second = storage.save_with_images(
        image, result, sources_dir=tmp_path, crops_dir=tmp_path
    )
    assert first != second
    a, b = storage.get_image_files(first), storage.get_image_files(second)
    assert a.source_path != b.source_path
    assert a.source_path.read_bytes() == b.source_path.read_bytes()


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 0, 31, 20),
        (-1, 0, 10, 10),
        (1, 1, 1, 5),
        (10, 2, 1, 3),
        (0.5, 1, 4, 5),
        (True, 1, 4, 5),
    ],
)
def test_invalid_crop_bounds_write_nothing(tmp_path, result, image, bounds):
    result = replace(result, plates=[replace(result.plates[0], crop_bbox_xyxy=bounds)])
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    with pytest.raises(ValueError, match="crop_bbox_xyxy"):
        storage.save_with_images(
            image, result, sources_dir=tmp_path / "s", crops_dir=tmp_path / "c"
        )
    assert storage.get(1) is None
    assert not (tmp_path / "s").exists() and not (tmp_path / "c").exists()


def test_image_dimension_mismatch_writes_nothing(tmp_path, result, image):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    with pytest.raises(ValueError, match="image_size_wh"):
        storage.save_with_images(
            image,
            replace(result, image_size_wh=(20, 30)),
            sources_dir=tmp_path / "s",
            crops_dir=tmp_path / "c",
        )
    assert storage.get(1) is None
    assert not (tmp_path / "s").exists()


def test_png_incompatible_mode_is_rejected_without_pixel_conversion(tmp_path, result):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    with Image.new("F", (30, 20), -1.5) as image:
        with pytest.raises(ValueError, match="losslessly"):
            storage.save_with_images(
                image, result, sources_dir=tmp_path / "s", crops_dir=tmp_path / "c"
            )
        assert image.getpixel((0, 0)) == -1.5
    assert storage.get(1) is None


def assert_no_image_save(storage, tmp_path):
    with closing(sqlite3.connect(storage.database_path)) as connection:
        for table in ("observations", "observation_files"):
            assert (
                connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
            )
    for name in ("s", "c"):
        root = tmp_path / name
        assert not root.exists() or list(root.iterdir()) == []


@pytest.mark.parametrize("filename", ["source.png", "plate_001.png"])
def test_partial_file_failure_cleans_folders_and_rolls_back(
    tmp_path, result, image, monkeypatch, filename
):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    real_save = Image.Image.save

    def fail_save(pixels, path, *args, **kwargs):
        if path.name == filename:
            path.write_bytes(b"partially written PNG")
            raise OSError("disk full")
        return real_save(pixels, path, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(OSError, match=filename):
        storage.save_with_images(
            image, result, sources_dir=tmp_path / "s", crops_dir=tmp_path / "c"
        )
    assert_no_image_save(storage, tmp_path)


@pytest.mark.parametrize("stage", ["insert", "commit"])
def test_database_failure_after_files_rolls_back_both_rows(
    tmp_path, result, image, monkeypatch, stage
):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    monkeypatch.setattr(module.sqlite3, "connect", connect)
    with closing(connect(storage.database_path)) as connection, connection:
        if stage == "insert":
            connection.execute("""
                CREATE TRIGGER reject_files BEFORE INSERT ON observation_files
                BEGIN SELECT RAISE(ABORT, 'forced file reference failure'); END
            """)
        else:
            connection.executescript("""
                CREATE TABLE parent (id INTEGER PRIMARY KEY);
                CREATE TABLE pending (
                    parent_id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED
                );
                CREATE TRIGGER defer_file_failure AFTER INSERT ON observation_files
                BEGIN INSERT INTO pending VALUES (999); END;
            """)
    with pytest.raises(sqlite3.IntegrityError):
        storage.save_with_images(
            image, result, sources_dir=tmp_path / "s", crops_dir=tmp_path / "c"
        )
    assert_no_image_save(storage, tmp_path)


def test_cleanup_failure_reports_original_error_and_residual_paths(
    tmp_path, result, image, monkeypatch
):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")

    def fail_save(pixels, path, **kwargs):
        path.write_bytes(b"partial")
        raise OSError("disk full")

    def fail_cleanup(path):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    monkeypatch.setattr(module.shutil, "rmtree", fail_cleanup)
    with pytest.raises(
        OSError, match="disk full.*cleanup failed.*cleanup denied"
    ) as error:
        storage.save_with_images(
            image, result, sources_dir=tmp_path / "s", crops_dir=tmp_path / "c"
        )
    assert str(tmp_path / "s") in str(error.value)
    assert isinstance(error.value.__cause__, OSError)
    assert storage.get(1) is None
    assert list((tmp_path / "s").rglob("source.png"))


def test_folder_collision_preserves_existing_files(
    tmp_path, result, image, monkeypatch
):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    monkeypatch.setattr(module, "uuid4", lambda: MagicMock(hex="existing"))
    folder = tmp_path / "s" / "existing"
    folder.mkdir(parents=True)
    sentinel = folder / "source.png"
    sentinel.write_bytes(b"existing user work")
    with pytest.raises(FileExistsError):
        storage.save_with_images(
            image, result, sources_dir=tmp_path / "s", crops_dir=tmp_path / "c"
        )
    assert sentinel.read_bytes() == b"existing user work"
    assert storage.get(1) is None


@pytest.mark.parametrize("observation_id", [None, True, 0, -1, "1", 2**63])
def test_image_reference_id_validation(tmp_path, observation_id):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    with pytest.raises(ValueError, match="observation_id"):
        storage.get_image_files(observation_id)


@pytest.mark.parametrize("payload", ["not JSON", "{}", '["relative.png"]', "[]"])
def test_malformed_image_references_raise(tmp_path, result, image, payload):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    observation_id = storage.save_with_images(
        image, result, sources_dir=tmp_path / "s", crops_dir=tmp_path / "c"
    )
    with closing(sqlite3.connect(storage.database_path)) as connection, connection:
        connection.execute(
            "UPDATE observation_files SET crop_paths_json = ?", (payload,)
        )
    with pytest.raises(ValueError, match="image references"):
        storage.get_image_files(observation_id)


def test_registered_missing_file_is_an_error(tmp_path, result, image):
    storage = module.SQLiteStorage(tmp_path / "observations.sqlite3")
    observation_id = storage.save_with_images(
        image, result, sources_dir=tmp_path / "s", crops_dir=tmp_path / "c"
    )
    files = storage.get_image_files(observation_id)
    files.crop_paths[1].unlink()
    with pytest.raises(FileNotFoundError, match="plate_001"):
        storage.get_image_files(observation_id)
