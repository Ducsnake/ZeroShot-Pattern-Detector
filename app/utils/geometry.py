from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def as_int(self) -> "BBox":
        return BBox(round(self.x1), round(self.y1), round(self.x2), round(self.y2))

    def clip(self, width: int, height: int) -> "BBox":
        return BBox(
            float(np.clip(self.x1, 0, max(0, width - 1))),
            float(np.clip(self.y1, 0, max(0, height - 1))),
            float(np.clip(self.x2, 0, width)),
            float(np.clip(self.y2, 0, height)),
        )

    def scale(self, sx: float, sy: float | None = None) -> "BBox":
        if sy is None:
            sy = sx
        return BBox(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)

    def to_dict(self) -> dict[str, float]:
        return {
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
            "width": float(self.width),
            "height": float(self.height),
        }

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> "BBox":
        return BBox(float(x), float(y), float(x + w), float(y + h))


@dataclass
class Detection:
    bbox: BBox
    score: float
    source: str
    label: str = "pattern"
    metadata: dict[str, Any] = field(default_factory=dict)

    def clipped(self, width: int, height: int) -> "Detection":
        return Detection(
            bbox=self.bbox.clip(width, height),
            score=float(self.score),
            source=self.source,
            label=self.label,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "bbox": self.bbox.to_dict(),
            "score": float(self.score),
            "source": self.source,
            "label": self.label,
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data


def iou(box_a: BBox, box_b: BBox) -> float:
    x1 = max(box_a.x1, box_b.x1)
    y1 = max(box_a.y1, box_b.y1)
    x2 = min(box_a.x2, box_b.x2)
    y2 = min(box_a.y2, box_b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_a.area + box_b.area - inter
    return 0.0 if union <= 0 else float(inter / union)


def bbox_from_polygon(points: np.ndarray) -> BBox:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return BBox(float(x1), float(y1), float(x2), float(y2))


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def detections_to_jsonable(detections: list[Detection]) -> list[dict[str, Any]]:
    return [det.to_dict() for det in detections]

