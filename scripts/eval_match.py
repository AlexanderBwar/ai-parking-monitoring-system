"""Match saved plate detections to ground truth for quick error analysis."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from parking_monitoring.config import ConfigError, load_config

IOU_THRESH = 0.5
FN_LOW_CONF_FLOOR = 0.10
ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT / "runs/baseline-001/eval-epoch19/fresh-evaluation/predictions.json"
GT_PATH = ROOT / "data/raw/annotations/kaggle_val_annotations.json"
DEFAULT_ANALYSIS_DIR = ROOT / "runs/baseline-001/eval-epoch19/error-analysis-mvp"
COLUMNS = [
    "image_id", "file_name", "type", "fp_type", "score",
    "pred_x", "pred_y", "pred_w", "pred_h",
    "gt_x", "gt_y", "gt_w", "gt_h", "iou", "box_area_ratio",
    "best_overlap_score", "fn_type",
]


def xywh_to_xyxy(box):
    """Convert a box to corners without clipping it to the image."""
    x, y, width, height = box
    return x, y, x + width, y + height


def box_iou(first, second):
    """Return intersection over union for two XYWH boxes."""
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = xywh_to_xyxy(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def make_row(image, kind, prediction=None, gt_box=None, overlap="", fp_type=""):
    """Build one CSV row, retaining original XYWH coordinates."""
    row = dict.fromkeys(COLUMNS, "")
    row.update(image_id=image["id"], file_name=image["file_name"],
               type=kind, fp_type=fp_type, iou=overlap)
    if prediction is not None:
        row["score"] = prediction["score"]
        row.update(zip(("pred_x", "pred_y", "pred_w", "pred_h"), prediction["bbox"]))
    if gt_box is not None:
        row.update(zip(("gt_x", "gt_y", "gt_w", "gt_h"), gt_box))
    area_box = prediction["bbox"] if kind == "FP" else gt_box
    row["box_area_ratio"] = (
        area_box[2] * area_box[3] / (image["width"] * image["height"])
    )
    return row


def match_image(image, gt_boxes, predictions, *, score_threshold: float):
    """Greedily match one image; score ties retain prediction-file order."""
    all_predictions = predictions
    predictions = sorted(
        (p for p in predictions if p["score"] > score_threshold),
        key=lambda p: p["score"], reverse=True,
    )
    matched = set()
    rows = []
    for prediction in predictions:
        overlaps = [box_iou(prediction["bbox"], box) for box in gt_boxes]
        best_any = max(overlaps) if overlaps else ""
        unmatched = [index for index in range(len(gt_boxes)) if index not in matched]
        # Equal-IoU ties retain the ground-truth file order.
        best = max(unmatched, key=lambda index: overlaps[index]) if unmatched else None
        if best is not None and overlaps[best] >= IOU_THRESH:
            matched.add(best)
            rows.append(make_row(image, "TP", prediction, gt_boxes[best], best_any))
        else:
            if any(overlaps[index] >= IOU_THRESH for index in matched):
                fp_type = "duplicate"
            elif overlaps and 0.1 <= best_any < IOU_THRESH:
                fp_type = "poor_localization"
            else:
                fp_type = "background"
            rows.append(make_row(image, "FP", prediction, overlap=best_any, fp_type=fp_type))

    for index, gt_box in enumerate(gt_boxes):
        if index not in matched:
            # An FN has no prediction, so its score, prediction box and IoU are blank.
            row = make_row(image, "FN", gt_box=gt_box)
            best_overlap_score = max(
                (p["score"] for p in all_predictions
                 if box_iou(p["bbox"], gt_box) >= IOU_THRESH),
                default=None,
            )
            row["best_overlap_score"] = (
                best_overlap_score if best_overlap_score is not None else ""
            )
            if best_overlap_score is not None and best_overlap_score > score_threshold:
                row["fn_type"] = "matched_elsewhere"
            elif best_overlap_score is not None and best_overlap_score >= FN_LOW_CONF_FLOOR:
                row["fn_type"] = "low_confidence"
            else:
                row["fn_type"] = "not_detected"
            rows.append(row)
    return rows


def self_test():
    """Check the requested hand-made cases with the real matching function."""
    image = {"id": 1, "file_name": "example.jpg", "width": 100, "height": 100}
    gt = [[10, 10, 20, 20]]
    prediction = {"bbox": [10, 10, 20, 20], "score": 0.9}
    threshold = 0.5

    rows = match_image(image, gt, [prediction], score_threshold=threshold)
    assert [row["type"] for row in rows] == ["TP"]
    assert rows[0]["iou"] == 1.0

    rows = match_image(
        image, gt, [prediction, {**prediction, "score": 0.8}], score_threshold=threshold
    )
    assert [row["type"] for row in rows] == ["TP", "FP"]
    assert rows[1]["fp_type"] == "duplicate"

    rows = match_image(image, gt, [], score_threshold=threshold)
    assert [row["type"] for row in rows] == ["FN"]
    assert rows[0]["fn_type"] == "not_detected"
    assert rows[0]["best_overlap_score"] == ""
    rows = match_image(image, [], [prediction], score_threshold=threshold)
    assert [row["type"] for row in rows] == ["FP"]
    assert rows[0]["fp_type"] == "background" and rows[0]["iou"] == ""
    assert match_image(image, [], [], score_threshold=threshold) == []

    rows = match_image(
        image, gt, [{**prediction, "score": threshold}], score_threshold=threshold
    )
    assert [row["type"] for row in rows] == ["FN"]
    assert rows[0]["fn_type"] == "low_confidence"
    assert rows[0]["best_overlap_score"] == threshold
    rows = match_image(
        image, gt, [{**prediction, "score": 0.51}], score_threshold=threshold
    )
    assert [row["type"] for row in rows] == ["TP"]


def main():
    """Load, match, check totals, and save the requested CSV and summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR,
                        help="Directory for new matches.csv and summary.json")
    analysis_dir = parser.parse_args().analysis_dir.expanduser()
    matches_path = analysis_dir / "matches.csv"
    summary_path = analysis_dir / "summary.json"
    review_path = analysis_dir / "review"
    for path in (matches_path, summary_path, review_path):
        if path.exists() or path.is_symlink():
            parser.error(f"{path} already exists. Choose a fresh --analysis-dir.")
    try:
        score_threshold = load_config().detector.score_threshold
    except ConfigError as exc:
        parser.error(str(exc))

    with GT_PATH.open(encoding="utf-8") as file:
        gt = json.load(file)
    with PREDICTIONS_PATH.open(encoding="utf-8") as file:
        predictions = json.load(file)

    images = {image["id"]: image for image in gt["images"]}
    gt_by_image = defaultdict(list)
    predictions_by_image = defaultdict(list)
    for annotation in gt["annotations"]:
        assert annotation["image_id"] in images, "Unknown GT image ID"
        gt_by_image[annotation["image_id"]].append(annotation["bbox"])
    for prediction in predictions:
        assert prediction["image_id"] in images, "Unknown prediction image ID"
        predictions_by_image[prediction["image_id"]].append(prediction)

    rows = []
    for image_id, image in images.items():
        rows.extend(match_image(
            image, gt_by_image[image_id], predictions_by_image[image_id],
            score_threshold=score_threshold,
        ))

    counts = Counter(row["type"] for row in rows)
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    retained_count = sum(p["score"] > score_threshold for p in predictions)
    assert tp + fn == len(gt["annotations"]), "TP + FN must equal the number of GT boxes"
    assert tp + fp == retained_count, "TP + FP must equal retained predictions"

    fp_counts = Counter(row["fp_type"] for row in rows if row["type"] == "FP")
    fn_counts = Counter(row["fn_type"] for row in rows if row["type"] == "FN")
    fp_type_counts = {
        kind: fp_counts[kind]
        for kind in ("duplicate", "poor_localization", "background")
    }
    fn_type_counts = {
        kind: fn_counts[kind]
        for kind in ("matched_elsewhere", "low_confidence", "not_detected")
    }
    assert sum(fp_type_counts.values()) == fp
    assert sum(fn_type_counts.values()) == fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "F1": f1,
        "score_threshold": score_threshold, "iou_threshold": IOU_THRESH,
        "fp_type_counts": fp_type_counts,
        "fn_type_counts": fn_type_counts,
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    try:
        with matches_path.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with summary_path.open("x", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, allow_nan=False)
            file.write("\n")
    except FileExistsError as exc:
        parser.error(f"{exc.filename} already exists. Choose a fresh --analysis-dir.")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    self_test()
    main()
