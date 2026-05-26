from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from skimage.feature import hog

from app.configs.config import DeepConfig
from app.models.dinov2 import DinoV2Model
from app.preprocess import ensure_gray
from app.utils.exceptions import ModelLoadError
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)


def image_fingerprint(image: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(image.shape).encode("utf-8"))
    h.update(np.ascontiguousarray(image).data)
    return h.hexdigest()


class BaseEmbeddingExtractor(ABC):
    name: str

    @abstractmethod
    def embed_batch(self, patches: list[np.ndarray]) -> torch.Tensor:
        """Return L2-normalized embeddings with shape [N, D]."""

    def embed_one(self, patch: np.ndarray) -> torch.Tensor:
        return self.embed_batch([patch])[0]


@dataclass
class LineArtEmbeddingExtractor(BaseEmbeddingExtractor):
    input_size: int = 96
    name: str = "lineart-hog"

    def _prepare(self, patch: np.ndarray) -> np.ndarray:
        gray = ensure_gray(patch)
        if gray.size == 0:
            gray = np.full((self.input_size, self.input_size), 255, dtype=np.uint8)
        resized = cv2.resize(gray, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        if float(resized.mean()) < 127.0:
            resized = cv2.bitwise_not(resized)
        return resized

    def _descriptor(self, patch: np.ndarray) -> np.ndarray:
        img = self._prepare(patch)
        foreground = (255 - img).astype(np.float32) / 255.0
        low_res = cv2.resize(foreground, (24, 24), interpolation=cv2.INTER_AREA).reshape(-1)
        h_profile = foreground.mean(axis=0)
        v_profile = foreground.mean(axis=1)
        edges = cv2.Canny(img, 40, 120)
        edge_low = cv2.resize(edges.astype(np.float32) / 255.0, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
        hog_features = hog(
            img,
            orientations=9,
            pixels_per_cell=(12, 12),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)
        descriptor = np.concatenate([low_res, h_profile, v_profile, edge_low, hog_features]).astype(np.float32)
        norm = np.linalg.norm(descriptor)
        if norm < 1e-8:
            return descriptor
        return descriptor / norm

    def embed_batch(self, patches: list[np.ndarray]) -> torch.Tensor:
        if not patches:
            return torch.empty((0, 1), dtype=torch.float32)
        features = np.stack([self._descriptor(patch) for patch in patches], axis=0)
        tensor = torch.from_numpy(features).float()
        return torch.nn.functional.normalize(tensor, dim=1)


class DinoV2EmbeddingExtractor(BaseEmbeddingExtractor):
    name = "dinov2"

    def __init__(self, config: DeepConfig) -> None:
        self.config = config
        self.input_size = int(config.input_size)
        self.model = DinoV2Model(
            model_name=config.model_name,
            device=config.device,
            allow_download=config.allow_dinov2_download,
        )
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def _preprocess_batch(self, patches: list[np.ndarray]) -> torch.Tensor:
        batch: list[np.ndarray] = []
        for patch in patches:
            gray = ensure_gray(patch)
            resized = cv2.resize(gray, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            rgb = np.repeat(resized[:, :, None], 3, axis=2).astype(np.float32) / 255.0
            batch.append(np.transpose(rgb, (2, 0, 1)))
        tensor = torch.from_numpy(np.stack(batch, axis=0)).float()
        return (tensor - self._mean) / self._std

    def embed_batch(self, patches: list[np.ndarray]) -> torch.Tensor:
        if not patches:
            return torch.empty((0, 1), dtype=torch.float32)
        tensor = self._preprocess_batch(patches)
        return self.model.encode(tensor)


class CompositeEmbeddingExtractor(BaseEmbeddingExtractor):
    def __init__(self, primary: BaseEmbeddingExtractor, fallback: BaseEmbeddingExtractor) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+fallback"
        self._primary_available = True

    def embed_batch(self, patches: list[np.ndarray]) -> torch.Tensor:
        if self._primary_available:
            try:
                return self.primary.embed_batch(patches)
            except ModelLoadError as exc:
                LOGGER.warning("DINOv2 unavailable; using %s fallback: %s", self.fallback.name, exc)
                self._primary_available = False
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                LOGGER.warning("DINOv2 embedding failed; using %s fallback: %s", self.fallback.name, exc)
                self._primary_available = False
        return self.fallback.embed_batch(patches)


def build_embedding_extractor(config: DeepConfig) -> BaseEmbeddingExtractor:
    fallback = LineArtEmbeddingExtractor(input_size=min(128, max(64, config.input_size // 2)))
    if not config.use_dinov2:
        return fallback
    return CompositeEmbeddingExtractor(DinoV2EmbeddingExtractor(config), fallback)

