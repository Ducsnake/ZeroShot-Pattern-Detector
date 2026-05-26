from __future__ import annotations

from dataclasses import dataclass

from app.utils.geometry import BBox, Detection, iou


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    mean_iou: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "mean_iou": self.mean_iou,
        }


def parse_gt_boxes(items: list[dict]) -> list[BBox]:
    boxes: list[BBox] = []
    for item in items:
        if "bbox" in item:
            box = item["bbox"]
        else:
            box = item
        if {"x1", "y1", "x2", "y2"}.issubset(box):
            boxes.append(BBox(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])))
        elif {"x", "y", "width", "height"}.issubset(box):
            boxes.append(BBox.from_xywh(float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"])))
    return boxes


def evaluate_detections(
    detections: list[Detection],
    ground_truth: list[BBox],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    sorted_detections = sorted(detections, key=lambda det: det.score, reverse=True)
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    matched_ious: list[float] = []

    for det in sorted_detections:
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue
            overlap = iou(det.bbox, gt)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return DetectionMetrics(precision, recall, f1, tp, fp, fn, mean_iou)

