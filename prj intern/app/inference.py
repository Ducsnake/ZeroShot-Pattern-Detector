from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from app.configs.config import AppConfig, load_config
from app.matching.classical_matcher import ClassicalFeatureMatcher
from app.matching.deep_matcher import DeepSimilarityMatcher
from app.postprocess import PostProcessor
from app.preprocess import preprocess_for_matching, resize_max_side
from app.utils.geometry import BBox, Detection, detections_to_jsonable
from app.utils.io import ImageLike, export_detections_json, load_image, save_image
from app.utils.logging import get_logger, setup_logging
from app.visualization.draw import draw_detections


LOGGER = get_logger(__name__)


@dataclass
class InferenceResult:
    detections: list[Detection]
    output_image: np.ndarray
    pattern_shape: tuple[int, int]
    drawing_shape: tuple[int, int]
    timings: dict[str, float] = field(default_factory=dict)
    heatmap: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": detections_to_jsonable(self.detections),
            "num_detections": len(self.detections),
            "pattern_shape": self.pattern_shape,
            "drawing_shape": self.drawing_shape,
            "timings": self.timings,
        }


class PatternDetector:
    def __init__(self, config_path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> None:
        self.config: AppConfig = load_config(config_path, overrides=overrides)
        setup_logging(self.config.inference.log_level)
        self.classical_matcher = ClassicalFeatureMatcher(self.config.classical)
        self.deep_matcher = DeepSimilarityMatcher(self.config.deep)
        self.postprocessor = PostProcessor(self.config.postprocess)

    def predict(
        self,
        pattern_image: ImageLike,
        drawing_image: ImageLike,
        *,
        enable_classical: bool | None = None,
        enable_deep: bool | None = None,
    ) -> InferenceResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}

        pattern_gray = load_image(pattern_image, force_rgb=False)
        drawing_rgb = load_image(drawing_image, force_rgb=True)
        drawing_gray = load_image(drawing_image, force_rgb=False)

        t0 = time.perf_counter()
        pattern_resized = resize_max_side(pattern_gray, max(512, self.config.preprocess.max_image_side // 2))
        drawing_resized = resize_max_side(drawing_gray, self.config.preprocess.max_image_side)
        pattern_pre = preprocess_for_matching(pattern_resized.image, self.config.preprocess)
        drawing_pre = preprocess_for_matching(drawing_resized.image, self.config.preprocess)
        timings["preprocess_seconds"] = time.perf_counter() - t0

        detections_resized: list[Detection] = []
        run_classical = self.config.inference.enable_classical if enable_classical is None else enable_classical
        run_deep = self.config.inference.enable_deep if enable_deep is None else enable_deep

        if run_classical and self.config.classical.enabled:
            t0 = time.perf_counter()
            detections_resized.extend(self.classical_matcher.match(pattern_pre, drawing_pre))
            timings["classical_seconds"] = time.perf_counter() - t0
        else:
            timings["classical_seconds"] = 0.0

        elapsed = time.perf_counter() - started
        if run_deep and self.config.deep.enabled and elapsed < self.config.inference.timeout_seconds:
            t0 = time.perf_counter()
            detections_resized.extend(self.deep_matcher.match(pattern_pre, drawing_pre))
            timings["deep_seconds"] = time.perf_counter() - t0
        else:
            timings["deep_seconds"] = 0.0

        t0 = time.perf_counter()
        detections_original = self._scale_detections(
            detections_resized,
            sx=drawing_resized.scale_x,
            sy=drawing_resized.scale_y,
        )
        detections = self.postprocessor.run(detections_original, drawing_rgb.shape[:2])
        timings["postprocess_seconds"] = time.perf_counter() - t0
        timings["total_seconds"] = time.perf_counter() - started

        rendered = draw_detections(drawing_rgb, detections)
        return InferenceResult(
            detections=detections,
            output_image=rendered,
            pattern_shape=pattern_gray.shape[:2],
            drawing_shape=drawing_rgb.shape[:2],
            timings=timings,
            heatmap=self.deep_matcher.last_heatmap,
        )

    async def predict_async(
        self,
        pattern_image: ImageLike,
        drawing_image: ImageLike,
        *,
        enable_classical: bool | None = None,
        enable_deep: bool | None = None,
    ) -> InferenceResult:
        return await asyncio.to_thread(
            self.predict,
            pattern_image,
            drawing_image,
            enable_classical=enable_classical,
            enable_deep=enable_deep,
        )

    def batch_predict(self, patterns: list[ImageLike], drawing_image: ImageLike) -> list[InferenceResult]:
        return [self.predict(pattern, drawing_image) for pattern in patterns]

    def export(
        self,
        result: InferenceResult,
        output_dir: str | Path = "outputs",
        prefix: str = "result",
    ) -> tuple[Path, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        image_path = save_image(output / f"{prefix}.png", result.output_image)
        json_path = export_detections_json(output / f"{prefix}.json", result.detections, extra={"timings": result.timings})
        return image_path, json_path

    @staticmethod
    def _scale_detections(detections: list[Detection], sx: float, sy: float) -> list[Detection]:
        scaled: list[Detection] = []
        for det in detections:
            metadata = dict(det.metadata)
            metadata["bbox_resized"] = det.bbox.to_dict()
            scaled.append(
                Detection(
                    bbox=BBox(det.bbox.x1 * sx, det.bbox.y1 * sy, det.bbox.x2 * sx, det.bbox.y2 * sy),
                    score=det.score,
                    source=det.source,
                    label=det.label,
                    metadata=metadata,
                )
            )
        return scaled

