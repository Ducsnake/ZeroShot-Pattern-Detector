from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.configs.config import ClassicalConfig
from app.feature_extractors.classical import FeatureSet, extract_feature_sets, match_descriptors
from app.preprocess import build_image_pyramid, crop_foreground, rotate_image_keep_bounds
from app.utils.geometry import BBox, Detection, bbox_from_polygon, polygon_area
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class _PatternVariant:
    image: np.ndarray
    angle: float
    scale: float
    feature_sets: list[FeatureSet]


class ClassicalFeatureMatcher:
    def __init__(self, config: ClassicalConfig) -> None:
        self.config = config

    def _pattern_variants(self, pattern: np.ndarray) -> list[_PatternVariant]:
        variants: list[_PatternVariant] = []
        base = crop_foreground(pattern, padding=6)
        for scale, scaled in build_image_pyramid(base, self.config.scales):
            for angle in self.config.rotations:
                rotated = rotate_image_keep_bounds(scaled, angle)
                rotated = crop_foreground(rotated, padding=4)
                if min(rotated.shape[:2]) < 8:
                    continue
                feature_sets = extract_feature_sets(rotated, self.config)
                if feature_sets:
                    variants.append(_PatternVariant(rotated, float(angle), float(scale), feature_sets))
        return variants

    def match(self, pattern: np.ndarray, drawing: np.ndarray) -> list[Detection]:
        target_features = extract_feature_sets(drawing, self.config)
        if not target_features:
            LOGGER.info("No classical features found in drawing")
            return []

        detections: list[Detection] = []
        target_by_name = {feature.name: feature for feature in target_features}
        variants = self._pattern_variants(pattern)
        if not variants:
            LOGGER.info("No classical features found in pattern variants")
            return []

        for variant in variants:
            for query_features in variant.feature_sets:
                target_features_for_type = target_by_name.get(query_features.name)
                if target_features_for_type is None:
                    continue
                matches = match_descriptors(query_features, target_features_for_type, self.config.ratio_test)
                if len(matches) < self.config.min_matches:
                    continue
                detections.extend(
                    self._extract_instances(
                        variant=variant,
                        query_features=query_features,
                        target_features=target_features_for_type,
                        matches=matches,
                        drawing_shape=drawing.shape[:2],
                    )
                )
        return detections

    def _extract_instances(
        self,
        variant: _PatternVariant,
        query_features: FeatureSet,
        target_features: FeatureSet,
        matches: list[cv2.DMatch],
        drawing_shape: tuple[int, int],
    ) -> list[Detection]:
        remaining = list(matches)
        detections: list[Detection] = []
        height, width = drawing_shape
        instance_idx = 0

        while len(remaining) >= self.config.min_matches and instance_idx < self.config.max_instances_per_variant:
            src = np.float32([query_features.keypoints[m.queryIdx].pt for m in remaining]).reshape(-1, 1, 2)
            dst = np.float32([target_features.keypoints[m.trainIdx].pt for m in remaining]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.config.ransac_reproj_threshold)
            if homography is None or mask is None:
                break

            inlier_mask = mask.ravel().astype(bool)
            inlier_count = int(inlier_mask.sum())
            if inlier_count < self.config.min_inliers:
                break

            h, w = variant.image.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
            bbox = bbox_from_polygon(projected).clip(width, height)
            if self._is_valid_geometry(projected, bbox, variant.image.shape[:2], drawing_shape):
                score = self._confidence(remaining, inlier_mask, projected, variant.image.shape[:2])
                detections.append(
                    Detection(
                        bbox=bbox,
                        score=score,
                        source=f"classical-{query_features.name}",
                        metadata={
                            "rotation": variant.angle,
                            "scale": variant.scale,
                            "inliers": inlier_count,
                            "matches": len(remaining),
                        },
                    )
                )
                instance_idx += 1

            remaining = [match for match, is_inlier in zip(remaining, inlier_mask) if not is_inlier]

        return detections

    def _is_valid_geometry(
        self,
        projected: np.ndarray,
        bbox: BBox,
        pattern_shape: tuple[int, int],
        drawing_shape: tuple[int, int],
    ) -> bool:
        if bbox.area < 16 or bbox.width < 4 or bbox.height < 4:
            return False
        drawing_h, drawing_w = drawing_shape
        if bbox.x1 >= drawing_w or bbox.y1 >= drawing_h or bbox.x2 <= 0 or bbox.y2 <= 0:
            return False
        projected_area = polygon_area(projected)
        pattern_area = float(pattern_shape[0] * pattern_shape[1])
        if pattern_area <= 0 or projected_area <= 0:
            return False
        area_ratio = projected_area / pattern_area
        return 0.05 <= area_ratio <= 25.0

    def _confidence(
        self,
        matches: list[cv2.DMatch],
        inlier_mask: np.ndarray,
        projected: np.ndarray,
        pattern_shape: tuple[int, int],
    ) -> float:
        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / max(1, len(matches))
        distances = np.array([match.distance for match, keep in zip(matches, inlier_mask) if keep], dtype=np.float32)
        if distances.size:
            distance_score = 1.0 / (1.0 + float(np.median(distances)) / 64.0)
        else:
            distance_score = 0.0
        projected_area = polygon_area(projected)
        pattern_area = float(pattern_shape[0] * pattern_shape[1])
        scale_score = min(projected_area, pattern_area) / max(projected_area, pattern_area, 1.0)
        match_score = min(1.0, inlier_count / 35.0)
        score = 0.25 + 0.30 * inlier_ratio + 0.25 * match_score + 0.15 * distance_score + 0.05 * scale_score
        return float(np.clip(score, 0.0, 1.0))

