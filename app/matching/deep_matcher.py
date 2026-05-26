from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from skimage.filters import gaussian

from app.configs.config import DeepConfig
from app.feature_extractors.deep import BaseEmbeddingExtractor, build_embedding_extractor, image_fingerprint
from app.preprocess import crop_foreground, resize_max_side, rotate_image_keep_bounds
from app.utils.cache import LRUCache
from app.utils.geometry import BBox, Detection
from app.utils.logging import get_logger
from app.utils.tiling import iter_tiles


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class Window:
    x: int
    y: int
    w: int
    h: int
    scale: float
    rotation: float


class DeepSimilarityMatcher:
    def __init__(self, config: DeepConfig, extractor: BaseEmbeddingExtractor | None = None) -> None:
        self.config = config
        self.extractor = extractor or build_embedding_extractor(config)
        self.last_heatmap: np.ndarray | None = None
        self._query_cache: LRUCache[str, list[tuple[float, np.ndarray, torch.Tensor]]] = LRUCache(max_items=24)

    def match(self, pattern: np.ndarray, drawing: np.ndarray) -> list[Detection]:
        scan = resize_max_side(drawing, self.config.max_scan_side)
        scan_image = scan.image
        scan_scale_x = scan_image.shape[1] / float(drawing.shape[1])
        scan_scale_y = scan_image.shape[0] / float(drawing.shape[0])
        query_base = crop_foreground(pattern, padding=4)

        detections: list[Detection] = []
        heatmap = np.zeros(scan_image.shape[:2], dtype=np.float32)
        heatmap_count = np.zeros(scan_image.shape[:2], dtype=np.float32)

        query_variants = self._query_variants(query_base)
        for rotation, query_image, query_embedding in query_variants:
            windows = self._generate_windows(query_image, scan_image.shape[:2], scan_scale_x, scan_scale_y, rotation)
            if not windows:
                continue
            detections.extend(
                self._score_windows(
                    query_embedding=query_embedding,
                    windows=windows,
                    scan_image=scan_image,
                    scale_back_x=scan.scale_x,
                    scale_back_y=scan.scale_y,
                    heatmap=heatmap,
                    heatmap_count=heatmap_count,
                )
            )

        self.last_heatmap = self._finalize_heatmap(heatmap, heatmap_count, drawing.shape[:2])
        return detections

    def _query_variants(self, query_base: np.ndarray) -> list[tuple[float, np.ndarray, torch.Tensor]]:
        cache_key = f"{self.extractor.name}:{tuple(self.config.rotations)}:{image_fingerprint(query_base)}"
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        variants: list[tuple[float, np.ndarray, torch.Tensor]] = []
        seen_shapes: set[tuple[int, int, int]] = set()
        for rotation in self.config.rotations:
            rotated = crop_foreground(rotate_image_keep_bounds(query_base, rotation), padding=2)
            if min(rotated.shape[:2]) < 4:
                continue
            shape_key = (int(rotation) % 360, rotated.shape[0], rotated.shape[1])
            if shape_key in seen_shapes:
                continue
            seen_shapes.add(shape_key)
            embedding = self.extractor.embed_one(rotated)
            variants.append((float(rotation), rotated, embedding))
        self._query_cache.put(cache_key, variants)
        return variants

    def _generate_windows(
        self,
        query_image: np.ndarray,
        scan_shape: tuple[int, int],
        scan_scale_x: float,
        scan_scale_y: float,
        rotation: float,
    ) -> list[Window]:
        scan_h, scan_w = scan_shape
        qh, qw = query_image.shape[:2]
        windows: list[Window] = []
        for scale in self.config.scales:
            win_w = max(1, int(round(qw * scale * scan_scale_x)))
            win_h = max(1, int(round(qh * scale * scan_scale_y)))
            if win_w < self.config.min_window or win_h < self.config.min_window:
                continue
            if win_w > scan_w or win_h > scan_h:
                continue
            step = max(4, int(round(min(win_w, win_h) * self.config.step_ratio)))
            tile_size = max(int(self.config.tile_size), win_w, win_h)
            overlap = max(win_w, win_h, int(round(tile_size * self.config.tile_overlap_ratio)))
            tiles = iter_tiles(scan_w, scan_h, tile_size, overlap)
            seen: set[tuple[int, int, int, int]] = set()
            for tile in tiles:
                if tile.width < win_w or tile.height < win_h:
                    continue
                max_x = tile.x2 - win_w
                max_y = tile.y2 - win_h
                xs = list(range(tile.x1, max_x + 1, step))
                ys = list(range(tile.y1, max_y + 1, step))
                if not xs or xs[-1] != max_x:
                    xs.append(max_x)
                if not ys or ys[-1] != max_y:
                    ys.append(max_y)
                for y in ys:
                    for x in xs:
                        key = (x, y, win_w, win_h)
                        if key in seen:
                            continue
                        seen.add(key)
                        windows.append(Window(x, y, win_w, win_h, float(scale), float(rotation)))

        if len(windows) > self.config.max_windows:
            keep = np.linspace(0, len(windows) - 1, self.config.max_windows, dtype=np.int64)
            windows = [windows[int(idx)] for idx in keep]
            LOGGER.debug("Subsampled deep windows for rotation %.1f to %d", rotation, len(windows))
        return windows

    def _score_windows(
        self,
        query_embedding: torch.Tensor,
        windows: list[Window],
        scan_image: np.ndarray,
        scale_back_x: float,
        scale_back_y: float,
        heatmap: np.ndarray,
        heatmap_count: np.ndarray,
    ) -> list[Detection]:
        detections: list[Detection] = []
        query = torch.nn.functional.normalize(query_embedding.reshape(1, -1).float(), dim=1)
        batch_size = max(1, int(self.config.batch_size))

        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start : start + batch_size]
            patches = [scan_image[item.y : item.y + item.h, item.x : item.x + item.w] for item in batch_windows]
            embeddings = self.extractor.embed_batch(patches)
            if embeddings.numel() == 0:
                continue
            similarities = torch.matmul(torch.nn.functional.normalize(embeddings.float(), dim=1), query.T).squeeze(1)
            scores = similarities.detach().cpu().numpy().astype(np.float32)
            for window, raw_score in zip(batch_windows, scores):
                score = float(np.clip(raw_score, 0.0, 1.0))
                self._accumulate_heatmap(heatmap, heatmap_count, window, score)
                if score < self.config.similarity_threshold:
                    continue
                bbox = BBox.from_xywh(
                    window.x * scale_back_x,
                    window.y * scale_back_y,
                    window.w * scale_back_x,
                    window.h * scale_back_y,
                )
                detections.append(
                    Detection(
                        bbox=bbox,
                        score=score,
                        source=f"deep-{self.extractor.name}",
                        metadata={
                            "rotation": window.rotation,
                            "scale": window.scale,
                            "similarity": score,
                        },
                    )
                )
        return detections

    @staticmethod
    def _accumulate_heatmap(heatmap: np.ndarray, heatmap_count: np.ndarray, window: Window, score: float) -> None:
        y1, y2 = window.y, window.y + window.h
        x1, x2 = window.x, window.x + window.w
        heatmap[y1:y2, x1:x2] += score
        heatmap_count[y1:y2, x1:x2] += 1.0

    def _finalize_heatmap(self, heatmap: np.ndarray, heatmap_count: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        valid = heatmap_count > 0
        result = np.zeros_like(heatmap, dtype=np.float32)
        result[valid] = heatmap[valid] / heatmap_count[valid]
        if self.config.heatmap_smoothing_sigma > 0:
            result = gaussian(result, sigma=self.config.heatmap_smoothing_sigma, preserve_range=True).astype(np.float32)
        if result.shape[:2] != output_shape:
            result = cv2.resize(result, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_LINEAR)
        max_value = float(result.max()) if result.size else 0.0
        if max_value > 0:
            result /= max_value
        return result
