from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage import exposure

from app.configs.config import PreprocessConfig


@dataclass(frozen=True)
class ResizeResult:
    image: np.ndarray
    scale_x: float
    scale_y: float


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_max_side(image: np.ndarray, max_side: int) -> ResizeResult:
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return ResizeResult(image.copy(), 1.0, 1.0)

    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, new_size, interpolation=interpolation)
    return ResizeResult(resized, width / float(new_size[0]), height / float(new_size[1]))


def normalize_background(gray: np.ndarray, invert_if_dark_foreground: bool = True) -> np.ndarray:
    arr = ensure_gray(gray)
    if invert_if_dark_foreground and float(arr.mean()) < 110.0:
        return cv2.bitwise_not(arr)
    return arr


def denoise_image(gray: np.ndarray, strength: int = 7) -> np.ndarray:
    strength = int(max(1, strength))
    return cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def adaptive_binarize(gray: np.ndarray, block_size: int = 31, c: int = 7) -> np.ndarray:
    block_size = max(3, int(block_size) | 1)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        int(c),
    )


def morphology_clean(binary: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    white_background = float(binary.mean()) > 127.0
    foreground = cv2.bitwise_not(binary) if white_background else binary
    closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.bitwise_not(cleaned) if white_background else cleaned


def enhance_edges(gray: np.ndarray) -> np.ndarray:
    arr = ensure_gray(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(arr)
    blurred = cv2.GaussianBlur(equalized, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(equalized, 1.45, blurred, -0.45, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_for_matching(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    gray = normalize_background(ensure_gray(image), config.invert_if_dark_foreground)
    if config.denoise:
        gray = denoise_image(gray, config.denoise_strength)
    if config.edge_enhance:
        gray = enhance_edges(gray)
    if config.adaptive_threshold:
        gray = adaptive_binarize(gray, config.adaptive_block_size, config.adaptive_c)
    if config.morphology:
        gray = morphology_clean(gray, config.morphology_kernel)
    return gray.astype(np.uint8, copy=False)


def build_image_pyramid(image: np.ndarray, scales: list[float]) -> list[tuple[float, np.ndarray]]:
    pyramid: list[tuple[float, np.ndarray]] = []
    height, width = image.shape[:2]
    for scale in scales:
        if scale <= 0:
            continue
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        pyramid.append((float(scale), cv2.resize(image, new_size, interpolation=interpolation)))
    return pyramid


def rotate_image_keep_bounds(image: np.ndarray, angle_degrees: float, border_value: int | tuple[int, int, int] = 255) -> np.ndarray:
    if abs(angle_degrees) % 360 < 1e-6:
        return image.copy()

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(round(height * sin + width * cos))
    new_height = int(round(height * cos + width * sin))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def resize_exact(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    interpolation = cv2.INTER_AREA if width < image.shape[1] or height < image.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(image, (max(1, width), max(1, height)), interpolation=interpolation)


def foreground_mask(gray_or_binary: np.ndarray) -> np.ndarray:
    gray = ensure_gray(gray_or_binary)
    if float(gray.mean()) > 127.0:
        return (gray < 220).astype(np.uint8) * 255
    return (gray > 35).astype(np.uint8) * 255


def crop_foreground(gray: np.ndarray, padding: int = 4) -> np.ndarray:
    mask = foreground_mask(gray)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return gray.copy()
    h, w = gray.shape[:2]
    x1 = max(0, int(xs.min()) - padding)
    x2 = min(w, int(xs.max()) + padding + 1)
    y1 = max(0, int(ys.min()) - padding)
    y2 = min(h, int(ys.max()) + padding + 1)
    return gray[y1:y2, x1:x2].copy()


def contrast_stretch(gray: np.ndarray) -> np.ndarray:
    arr = ensure_gray(gray)
    p2, p98 = np.percentile(arr, (2, 98))
    if p98 <= p2:
        return arr
    stretched = exposure.rescale_intensity(arr, in_range=(p2, p98), out_range=(0, 255))
    return stretched.astype(np.uint8)
