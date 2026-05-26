from __future__ import annotations

import math

from app.utils.geometry import Detection


def calibrate_detection(det: Detection) -> Detection:
    score = float(det.score)
    if det.source.startswith("deep-"):
        similarity = float(det.metadata.get("similarity", score))
        score = 1.0 / (1.0 + math.exp(-12.0 * (similarity - 0.78)))
        score = max(float(det.score), score * 0.98)
    elif det.source.startswith("classical-"):
        inliers = float(det.metadata.get("inliers", 0))
        score = min(1.0, 0.92 * score + 0.08 * min(1.0, inliers / 30.0))
    return Detection(bbox=det.bbox, score=score, source=det.source, label=det.label, metadata=dict(det.metadata))

