"""Focused checks for pipeline orchestration and input/failure boundaries."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from PIL import Image

import parking_monitoring.pipeline as module
from parking_monitoring.config import load_config
from parking_monitoring.schemas import Detection, OCRResult


@pytest.fixture
def components(monkeypatch):
    config = load_config("configs/mvp.yaml")
    detector_factory = Mock()
    ocr_factory = Mock()
    monkeypatch.setattr(module, "RTDETRDetector", detector_factory)
    monkeypatch.setattr(module, "PaddleOCREngine", ocr_factory)
    return config, detector_factory, ocr_factory


def assert_closed(crop):
    with pytest.raises(ValueError):
        crop.getpixel((0, 0))


def test_multiple_plates_preserve_results_and_aggregate_timings(
    components, monkeypatch
):
    config, detector_factory, ocr_factory = components
    pipeline = module.ParkingPipeline(config)
    detector_factory.return_value.predict.return_value = [
        Detection((1.2, 2.4, 9.1, 8.2), 0.9),
        Detection((10, 2, 18, 10), 0.8),
        Detection((20, 1, 29, 15), 0.7),
    ]
    readings = [
        OCRResult("YGN A-1234", "YGNA-1234", 0.95, "read"),
        OCRResult(None, None, None, "error", "Backend failure"),
        OCRResult("MDY B-5678", "MDYB-5678", 0.4, "uncertain"),
    ]
    ocr_factory.return_value.recognize.side_effect = readings
    ticks = iter(range(16))
    monkeypatch.setattr(module, "perf_counter", lambda: next(ticks))

    with Image.new("RGB", (30, 20), (30, 60, 90)) as image:
        before = image.tobytes()
        result = pipeline.process(image, "example.jpg")
        assert image.tobytes() == before
        assert image.getpixel((0, 0)) == (30, 60, 90)

    assert result.source_id == "example.jpg"
    assert result.image_size_wh == (30, 20)
    assert [plate.ocr for plate in result.plates] == readings
    assert all(plate.ocr is reading for plate, reading in zip(result.plates, readings))
    assert [plate.crop_bbox_xyxy for plate in result.plates] == [
        (1, 2, 10, 9),
        (10, 2, 18, 10),
        (20, 1, 29, 15),
    ]
    assert result.timings_ms == {
        "detection": 1000,
        "cropping": 3000,
        "ocr": 3000,
        "total": 15000,
    }
    for call in ocr_factory.return_value.recognize.call_args_list:
        assert_closed(call.args[0])
    assert result.model_info["detector_checkpoint"] == str(
        config.detector.checkpoint_path
    )
    assert result.model_info["ocr_model"] == "paddleocr/PP-OCRv4/en"
    result.model_info["ocr_model"] = "modified copy"
    assert pipeline.model_info["ocr_model"] != "modified copy"


def test_empty_results_reuse_models_and_do_not_call_ocr(components):
    config, detector_factory, ocr_factory = components
    pipeline = module.ParkingPipeline(config)
    detector_factory.return_value.predict.return_value = []
    with Image.new("RGB", (10, 10)) as image:
        for _ in range(2):
            result = pipeline.process(image, "empty.jpg")
            assert result.plates == []
            assert result.timings_ms["cropping"] == result.timings_ms["ocr"] == 0
    detector_factory.assert_called_once_with(config.detector)
    ocr_factory.assert_called_once_with(config.ocr)
    ocr_factory.return_value.recognize.assert_not_called()


@pytest.mark.parametrize("padding", [None, -1, True, 1.5])
def test_invalid_settings_rejected_before_model_loading(components, padding):
    config, detector_factory, ocr_factory = components
    invalid = None if padding is None else replace(config, crop_padding_pixels=padding)
    with pytest.raises(ValueError, match="config|crop_padding_pixels"):
        module.ParkingPipeline(invalid)
    detector_factory.assert_not_called()
    ocr_factory.assert_not_called()


@pytest.mark.parametrize("source_id", ["", "  ", None, 42])
def test_invalid_source_identifier_rejected_before_detection(components, source_id):
    config, detector_factory, _ = components
    pipeline = module.ParkingPipeline(config)
    with (
        Image.new("RGB", (10, 10)) as image,
        pytest.raises(ValueError, match="source_id"),
    ):
        pipeline.process(image, source_id)
    detector_factory.return_value.predict.assert_not_called()


@pytest.mark.parametrize("kind", ["not-image", "empty", "closed"])
def test_invalid_image_rejected_before_detection(components, kind):
    config, detector_factory, _ = components
    pipeline = module.ParkingPipeline(config)
    image = None
    if kind == "empty":
        image = Image.new("RGB", (0, 10))
    elif kind == "closed":
        image = Image.new("RGB", (10, 10))
        image.close()
    try:
        with pytest.raises(ValueError, match="image"):
            pipeline.process(image, "bad.jpg")
    finally:
        if image is not None:
            image.close()
    detector_factory.return_value.predict.assert_not_called()


@pytest.mark.parametrize("stage", ["detector", "crop", "ocr"])
def test_processing_exceptions_propagate_and_crops_close(components, stage):
    config, detector_factory, ocr_factory = components
    pipeline = module.ParkingPipeline(config)
    detector = detector_factory.return_value
    ocr = ocr_factory.return_value
    detector.predict.return_value = [Detection((1, 1, 8, 8), 0.9)]
    exception = RuntimeError(f"{stage} failure")
    if stage == "detector":
        detector.predict.side_effect = exception
    elif stage == "crop":
        detector.predict.return_value = [Detection((20, 20, 30, 30), 0.9)]
    else:
        ocr.recognize.side_effect = exception

    with Image.new("RGB", (10, 10)) as image:
        with pytest.raises(ValueError if stage == "crop" else RuntimeError) as caught:
            pipeline.process(image, "failed.jpg")
        if stage != "crop":
            assert caught.value is exception
        assert image.getpixel((0, 0)) == (0, 0, 0)
    if stage == "ocr":
        assert_closed(ocr.recognize.call_args.args[0])
    else:
        ocr.recognize.assert_not_called()
