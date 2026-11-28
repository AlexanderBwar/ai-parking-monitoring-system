"""Draw plate annotations on separate display copies."""

# 1. Import the image and result contracts.
import math

from PIL import Image, ImageDraw, ImageFont

from parking_monitoring.schemas import Detection, ImageResult, OCRResult, PlateResult

_STATUS_COLORS = {
    "read": (0, 160, 0),
    "uncertain": (230, 160, 0),
    "unreadable": (120, 120, 120),
    "error": (220, 0, 0),
}


# 2. Define the public visualization interface.
def annotate_image(
    image: Image.Image,
    result: ImageResult,
) -> Image.Image:
    """Return a separate RGB copy with detector boxes and OCR labels.

    Args:
        image: The same source pixels and orientation used for detection.
        result: Processing results with matching image dimensions.

    Returns:
        An independent image of the same dimensions. Preserve source
        pixels and result data, including both stored box types.

    Raises:
        ValueError: Invalid input, mismatched dimensions, unsupported
            image mode, unknown status, invalid score or detector box.

    Detector outlines are unpadded: round outward, clip, then subtract one
    from exclusive right/bottom edges for Pillow's inclusive rectangle.
    Neither stored detector nor padded crop coordinates change. Unknown
    statuses and invalid boxes fail explicitly rather than being skipped.

    Labels show normalized text (or '(no text)'), status and OCR score
    (or 'OCR n/a'); never show error_message. Use a scalable default font,
    sized from image height, and outlines sized from image width. Fit labels
    above, below, then inside the box; shrink/shorten plate text as needed.
    If the required status/score label cannot fit, draw only the box.

    Accept ordinary display modes 1/L/LA/P/RGB/RGBA. Reject 16-bit, signed
    integer and float modes instead of silently clipping intensities. Matching
    dimensions cannot prove source identity; the caller must supply the right
    pixels/orientation. No inference, database access or file writes occur.
    """
    # 3. Validate the source image, display mode and result dimensions.
    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError("image must be a nonempty Pillow image")
    if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
        raise ValueError(f"Unsupported display image mode: {image.mode!r}")
    if not isinstance(result, ImageResult):
        raise ValueError("result must be an ImageResult")  # noqa: TRY004
    if (
        not isinstance(result.image_size_wh, tuple)
        or len(result.image_size_wh) != 2
        or any(type(value) is not int for value in result.image_size_wh)
        or result.image_size_wh != image.size
    ):
        raise ValueError("image.size must match result.image_size_wh")
    if not isinstance(result.plates, list):
        raise ValueError("result.plates must be a list")  # noqa: TRY004

    # 4. Prepare validated detector outlines and labels without changing results.
    width, height = image.size
    annotations = []
    for index, plate in enumerate(result.plates):
        if (
            not isinstance(plate, PlateResult)
            or not isinstance(plate.detection, Detection)
            or not isinstance(plate.ocr, OCRResult)
        ):
            raise ValueError(f"Plate {index} must contain Detection and OCRResult")  # noqa: TRY004
        status = plate.ocr.status
        if not isinstance(status, str) or status not in _STATUS_COLORS:
            raise ValueError(f"Plate {index} has unknown OCR status: {status!r}")
        box = plate.detection.bbox_xyxy
        if (
            not isinstance(box, tuple)
            or len(box) != 4
            or any(
                type(value) not in (int, float)
                or (type(value) is float and not math.isfinite(value))
                for value in box
            )
        ):
            raise ValueError(
                f"Plate {index} detector box must contain four finite numbers"
            )
        x0, y0, x1, y1 = box
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"Plate {index} detector box must be ordered and nonempty")
        if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
            raise ValueError(f"Plate {index} detector box is fully outside the image")
        rectangle = (
            max(0, math.floor(x0)),
            max(0, math.floor(y0)),
            min(width, math.ceil(x1)) - 1,
            min(height, math.ceil(y1)) - 1,
        )
        text = plate.ocr.normalized_text
        if text is not None and not isinstance(text, str):
            raise ValueError(f"Plate {index} normalized_text must be a string or None")
        score = plate.ocr.score
        if score is not None and (
            type(score) not in (int, float) or not 0 <= score <= 1
        ):
            raise ValueError(
                f"Plate {index} OCR score must be finite and in [0, 1] or None"
            )
        suffix = (
            f" | {status} | OCR {score:.2f}"
            if score is not None
            else f" | {status} | OCR n/a"
        )
        annotations.append(
            (rectangle, _STATUS_COLORS[status], text or "(no text)", suffix)
        )

    # 5. Make an independent RGB canvas; an empty plate list returns this copy.
    try:
        canvas = image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read source image pixels: {exc}") from exc
    try:
        draw = ImageDraw.Draw(canvas)
        base_font_size = max(12, round(height * 0.03))
        line_width = max(1, round(width * 0.002))
        for rectangle, color, text, suffix in annotations:
            # 6. Draw the unpadded detector box using inclusive pixel edges.
            left, top, right, bottom = rectangle
            draw.rectangle(
                rectangle,
                outline=color,
                width=min(line_width, right - left + 1, bottom - top + 1),
            )

            # 7. Adapt existing scripts' scalable-font and textbbox fitting logic.
            label = text + suffix
            for size in range(base_font_size, 5, -1):
                font = ImageFont.load_default(size=size)
                padding = max(2, size // 6)
                bounds = draw.textbbox((0, 0), label, font=font)
                label_width = bounds[2] - bounds[0] + 2 * padding
                label_height = bounds[3] - bounds[1] + 2 * padding
                if label_width <= width and label_height <= height:
                    break
            else:
                # Shorten only displayed plate text, retaining status and score.
                while text:
                    text = text[:-1]
                    label = text + "..." + suffix
                    bounds = draw.textbbox((0, 0), label, font=font)
                    label_width = bounds[2] - bounds[0] + 2 * padding
                    label_height = bounds[3] - bounds[1] + 2 * padding
                    if label_width <= width and label_height <= height:
                        break
                else:
                    continue  # A tiny canvas cannot fit the required label.

            # 8. Place above, then below, then inside; keep the label on canvas.
            label_left = min(left, width - label_width)
            if top >= label_height:
                label_top = top - label_height
            elif bottom + 1 + label_height <= height:
                label_top = bottom + 1
            else:
                label_top = min(top, height - label_height)
            draw.rectangle(
                (
                    label_left,
                    label_top,
                    label_left + label_width - 1,
                    label_top + label_height - 1,
                ),
                fill=color,
            )
            luminance = sum(
                channel * weight
                for channel, weight in zip(color, (0.299, 0.587, 0.114), strict=True)
            )
            draw.text(
                (label_left + padding - bounds[0], label_top + padding - bounds[1]),
                label,
                font=font,
                fill=(0, 0, 0) if luminance > 150 else (255, 255, 255),
            )
        # 9. Caller owns the returned canvas and decides whether to display/save.
        return canvas
    except BaseException:
        canvas.close()
        raise
