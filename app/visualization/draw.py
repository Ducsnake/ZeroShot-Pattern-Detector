from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.utils.geometry import Detection, detections_to_jsonable
from app.utils.io import save_image, save_json


_PALETTE = [
    (220, 38, 38),
    (37, 99, 235),
    (22, 163, 74),
    (202, 138, 4),
    (147, 51, 234),
    (14, 165, 233),
]


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    canvas = _as_rgb(image)
    for idx, det in enumerate(detections):
        color = _PALETTE[idx % len(_PALETTE)]
        box = det.bbox.as_int()
        p1 = (int(box.x1), int(box.y1))
        p2 = (int(box.x2), int(box.y2))
        cv2.rectangle(canvas, p1, p2, color, thickness)
        if not show_labels:
            continue
        label = f"{idx + 1}: {det.score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y = max(0, p1[1] - text_h - baseline - 4)
        cv2.rectangle(canvas, (p1[0], y), (p1[0] + text_w + 8, y + text_h + baseline + 6), color, -1)
        cv2.putText(
            canvas,
            label,
            (p1[0] + 4, y + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    canvas = _as_rgb(image)
    if heatmap.shape[:2] != canvas.shape[:2]:
        heatmap = cv2.resize(heatmap, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(heatmap, 0, 1)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(canvas, 1.0 - alpha, heat, alpha, 0)


def save_visualization(
    image: np.ndarray,
    detections: list[Detection],
    output_image_path: str | Path,
    output_json_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    rendered = draw_detections(image, detections)
    image_path = save_image(output_image_path, rendered)
    json_path = None
    if output_json_path is not None:
        json_path = save_json(output_json_path, {"detections": detections_to_jsonable(detections)})
    return image_path, json_path


def detections_to_pretty_json(detections: list[Detection]) -> str:
    return json.dumps({"detections": detections_to_jsonable(detections)}, indent=2, ensure_ascii=False)

