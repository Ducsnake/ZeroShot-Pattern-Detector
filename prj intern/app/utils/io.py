from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.utils.exceptions import ImageLoadError
from app.utils.geometry import Detection, detections_to_jsonable


ImageLike = str | Path | np.ndarray | Image.Image


def load_image(image: ImageLike, force_rgb: bool = False) -> np.ndarray:
    loaded_with_cv2 = False
    if isinstance(image, np.ndarray):
        arr = image.copy()
    elif isinstance(image, Image.Image):
        if force_rgb:
            arr = np.array(image.convert("RGB"))
        elif image.mode == "1":
            arr = np.array(image.convert("L"))
        else:
            arr = np.array(image)
    else:
        path = Path(image)
        if not path.exists():
            raise ImageLoadError(f"Image path does not exist: {path}")
        flags = cv2.IMREAD_COLOR if force_rgb else cv2.IMREAD_UNCHANGED
        arr = cv2.imread(str(path), flags)
        if arr is None:
            raise ImageLoadError(f"Could not decode image: {path}")
        loaded_with_cv2 = True
        if force_rgb:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 4:
        if loaded_with_cv2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        arr = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        loaded_with_cv2 = False
    if force_rgb and arr.ndim == 3:
        return arr
    if arr.ndim == 3:
        if loaded_with_cv2:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    raise ImageLoadError(f"Unsupported image shape: {arr.shape}")


def save_image(path: str | Path, image: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arr = image
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(output), arr)
    if not ok:
        raise OSError(f"Could not save image to {output}")
    return output


def save_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return output


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def export_detections_json(path: str | Path, detections: list[Detection], extra: dict[str, Any] | None = None) -> Path:
    payload: dict[str, Any] = {"detections": detections_to_jsonable(detections)}
    if extra:
        payload.update(extra)
    return save_json(path, payload)
