from __future__ import annotations

from app.inference import InferenceResult, PatternDetector
from app.utils.io import ImageLike


def search_multiple_patterns(
    detector: PatternDetector,
    patterns: list[ImageLike],
    drawing: ImageLike,
) -> list[InferenceResult]:
    """Run zero-shot search for several independent query patterns."""

    results: list[InferenceResult] = []
    for pattern in patterns:
        results.append(detector.predict(pattern, drawing))
    return results

