from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.configs.config import TemplateConfig
from app.preprocess import crop_foreground, foreground_mask, resize_exact, rotate_image_keep_bounds
from app.utils.geometry import BBox, Detection
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TemplateVariant:
    image: np.ndarray
    binary: np.ndarray
    angle: float
    scale: float


class TemplateMatcher:
    """NCC matcher for binary/line-art technical symbols.

    ORB/SIFT can fail on low-texture electrical symbols. This matcher compares
    foreground masks directly and is useful for query crops from legends or
    schematic symbols.
    """

    def __init__(self, config: TemplateConfig) -> None:
        self.config = config

    def match(self, pattern: np.ndarray, drawing: np.ndarray) -> list[Detection]:
        drawing_fg = self._to_foreground_float(drawing)
        variants = self._variants(pattern)
        detections: list[Detection] = []

        for variant in variants:
            if variant.image.shape[0] > drawing_fg.shape[0] or variant.image.shape[1] > drawing_fg.shape[1]:
                continue
            result = cv2.matchTemplate(drawing_fg, variant.image, cv2.TM_CCOEFF_NORMED)
            detections.extend(self._detections_from_response(result, variant, drawing_fg))
        return detections

    def _variants(self, pattern: np.ndarray) -> list[TemplateVariant]:
        base = crop_foreground(pattern, padding=4)
        variants: list[TemplateVariant] = []
        seen: set[tuple[int, int, int]] = set()
        for angle in self.config.rotations:
            rotated = crop_foreground(rotate_image_keep_bounds(base, angle), padding=2)
            for scale in self.config.scales:
                width = max(1, int(round(rotated.shape[1] * scale)))
                height = max(1, int(round(rotated.shape[0] * scale)))
                if width < self.config.min_template_size or height < self.config.min_template_size:
                    continue
                resized = resize_exact(rotated, (width, height))
                template = self._to_foreground_float(resized)
                if float(template.std()) < 1e-4:
                    continue
                key = (int(angle) % 360, template.shape[0], template.shape[1])
                if key in seen:
                    continue
                seen.add(key)
                variants.append(TemplateVariant(template, template > 0.20, float(angle), float(scale)))
        return variants

    def _detections_from_response(
        self,
        response: np.ndarray,
        variant: TemplateVariant,
        drawing_fg: np.ndarray,
    ) -> list[Detection]:
        if response.size == 0:
            return []
        kernel_size = self.config.local_max_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        local_max = cv2.dilate(response, kernel)
        mask = (response >= self.config.threshold) & (response >= local_max - 1e-6)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return []

        scores = response[ys, xs]
        candidate_limit = max(self.config.max_candidates_per_variant * 250, 5000)
        order = np.argsort(scores)[::-1][:candidate_limit]
        scored_detections: list[Detection] = []
        h, w = variant.image.shape[:2]
        for idx in order:
            x = int(xs[idx])
            y = int(ys[idx])
            ncc = float(scores[idx])
            patch = drawing_fg[y : y + h, x : x + w]
            foreground_f1 = self._foreground_f1(variant.binary, patch > 0.20)
            if foreground_f1 < self.config.min_foreground_f1:
                continue
            score = float(np.clip(0.60 * ncc + 0.40 * foreground_f1, 0.0, 1.0))
            if score < self.config.min_combined_score:
                continue
            scored_detections.append(
                Detection(
                    bbox=BBox.from_xywh(x, y, w, h),
                    score=score,
                    source="template-ncc",
                    metadata={
                        "rotation": variant.angle,
                        "scale": variant.scale,
                        "ncc": ncc,
                        "foreground_f1": foreground_f1,
                    },
                )
            )
        scored_detections.sort(key=lambda det: det.score, reverse=True)
        return scored_detections[: self.config.max_candidates_per_variant]

    @staticmethod
    def _to_foreground_float(image: np.ndarray) -> np.ndarray:
        mask = foreground_mask(image)
        mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=0.4)
        return (mask.astype(np.float32) / 255.0).copy()

    @staticmethod
    def _foreground_f1(template_binary: np.ndarray, patch_binary: np.ndarray) -> float:
        if template_binary.shape != patch_binary.shape:
            return 0.0
        template_sum = int(template_binary.sum())
        patch_sum = int(patch_binary.sum())
        if template_sum == 0 or patch_sum == 0:
            return 0.0
        intersection = int(np.logical_and(template_binary, patch_binary).sum())
        precision = intersection / patch_sum
        recall = intersection / template_sum
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))
