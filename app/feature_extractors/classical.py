from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.configs.config import ClassicalConfig
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class FeatureSet:
    name: str
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None
    norm_type: int

    @property
    def is_valid(self) -> bool:
        return self.descriptors is not None and len(self.keypoints) > 0 and len(self.descriptors) > 0


def _create_sift(nfeatures: int) -> cv2.Feature2D | None:
    try:
        return cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.01, edgeThreshold=12)
    except Exception as exc:  # pragma: no cover - depends on OpenCV build
        LOGGER.warning("SIFT unavailable in this OpenCV build: %s", exc)
        return None


def _create_orb(nfeatures: int) -> cv2.Feature2D:
    return cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=12,
        edgeThreshold=8,
        patchSize=31,
        fastThreshold=7,
        scoreType=cv2.ORB_HARRIS_SCORE,
    )


def _extract(detector: cv2.Feature2D, image: np.ndarray, name: str, norm_type: int) -> FeatureSet:
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return FeatureSet(name=name, keypoints=tuple(keypoints or ()), descriptors=descriptors, norm_type=norm_type)


def extract_feature_sets(image: np.ndarray, config: ClassicalConfig) -> list[FeatureSet]:
    feature_sets: list[FeatureSet] = []
    if config.use_sift:
        sift = _create_sift(config.sift_features)
        if sift is not None:
            feature_sets.append(_extract(sift, image, "sift", cv2.NORM_L2))
    if config.use_orb:
        orb = _create_orb(config.orb_features)
        feature_sets.append(_extract(orb, image, "orb", cv2.NORM_HAMMING))
    return [features for features in feature_sets if features.is_valid]


def match_descriptors(
    query: FeatureSet,
    target: FeatureSet,
    ratio_test: float,
) -> list[cv2.DMatch]:
    if not query.is_valid or not target.is_valid or query.name != target.name:
        return []

    try:
        if query.norm_type == cv2.NORM_L2:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=64)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            query_desc = query.descriptors.astype(np.float32, copy=False)
            target_desc = target.descriptors.astype(np.float32, copy=False)
            knn = matcher.knnMatch(query_desc, target_desc, k=2)
        else:
            matcher = cv2.BFMatcher(query.norm_type, crossCheck=False)
            knn = matcher.knnMatch(query.descriptors, target.descriptors, k=2)
    except cv2.error as exc:
        LOGGER.debug("Descriptor matching failed for %s: %s", query.name, exc)
        return []

    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_test * second.distance:
            good.append(first)
    return sorted(good, key=lambda match: match.distance)
