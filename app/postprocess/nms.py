from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.configs.config import PostprocessConfig
from app.postprocess.calibration import calibrate_detection
from app.utils.geometry import BBox, Detection, iou


def non_max_suppression(detections: list[Detection], iou_threshold: float, max_detections: int) -> list[Detection]:
    if not detections:
        return []
    ordered = sorted(detections, key=lambda det: det.score, reverse=True)
    keep: list[Detection] = []
    while ordered and len(keep) < max_detections:
        current = ordered.pop(0)
        keep.append(current)
        ordered = [candidate for candidate in ordered if iou(current.bbox, candidate.bbox) <= iou_threshold]
    return keep


@dataclass
class PostProcessor:
    config: PostprocessConfig

    def run(self, detections: list[Detection], image_shape: tuple[int, int]) -> list[Detection]:
        height, width = image_shape
        filtered = [
            calibrate_detection(det).clipped(width, height)
            for det in detections
            if det.score >= self.config.score_threshold and det.bbox.area >= self.config.min_box_area
        ]
        if not filtered:
            return []
        merged = self._merge_duplicates(filtered)
        return non_max_suppression(merged, self.config.nms_iou_threshold, self.config.max_detections)

    def _merge_duplicates(self, detections: list[Detection]) -> list[Detection]:
        remaining = sorted(detections, key=lambda det: det.score, reverse=True)
        merged: list[Detection] = []

        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            survivors: list[Detection] = []
            for candidate in remaining:
                if iou(seed.bbox, candidate.bbox) >= self.config.duplicate_iou_threshold:
                    group.append(candidate)
                else:
                    survivors.append(candidate)
            remaining = survivors
            merged.append(self._merge_group(group))

        return merged

    @staticmethod
    def _merge_group(group: list[Detection]) -> Detection:
        if len(group) == 1:
            return group[0]

        weights = np.array([max(1e-3, det.score) for det in group], dtype=np.float32)
        coords = np.array([det.bbox.xyxy for det in group], dtype=np.float32)
        merged_coords = (coords * weights[:, None]).sum(axis=0) / weights.sum()
        sources = sorted({det.source for det in group})
        metadata = {
            "merged_count": len(group),
            "sources": sources,
            "members": [det.to_dict() for det in group[:8]],
        }
        score = min(1.0, max(det.score for det in group) + 0.04 * (len(sources) - 1) + 0.01 * min(5, len(group) - 1))
        return Detection(
            bbox=BBox(*[float(value) for value in merged_coords]),
            score=score,
            source="+".join(sources),
            label=group[0].label,
            metadata=metadata,
        )

