from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def draw_pattern(size: int = 112) -> Image.Image:
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    pad = size // 8
    center = size // 2
    outer = [pad, pad, size - pad, size - pad]
    inner = [size * 0.31, size * 0.31, size * 0.69, size * 0.69]
    draw.ellipse(outer, outline=0, width=max(2, size // 28))
    draw.ellipse(inner, outline=0, width=max(2, size // 32))
    draw.line([(center, pad), (center, size - pad)], fill=0, width=max(1, size // 45))
    draw.line([(pad, center), (size - pad, center)], fill=0, width=max(1, size // 45))
    hole_r = size // 16
    for x, y in [(size * 0.28, size * 0.28), (size * 0.72, size * 0.28), (size * 0.72, size * 0.72), (size * 0.28, size * 0.72)]:
        draw.ellipse([x - hole_r, y - hole_r, x + hole_r, y + hole_r], outline=0, width=max(1, size // 45))
    draw.rectangle([size * 0.44, pad - 4, size * 0.56, pad + size * 0.12], outline=0, width=max(1, size // 45))
    return img


def add_drawing_context(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> None:
    draw.rectangle([18, 18, width - 18, height - 18], outline=0, width=2)
    draw.rectangle([width - 260, height - 120, width - 18, height - 18], outline=0, width=1)
    for idx in range(5):
        y = height - 120 + idx * 20
        draw.line([(width - 260, y), (width - 18, y)], fill=80, width=1)
    for x in [width - 200, width - 130, width - 70]:
        draw.line([(x, height - 120), (x, height - 18)], fill=80, width=1)
    for _ in range(28):
        x1 = rng.randint(40, width - 320)
        y1 = rng.randint(40, height - 80)
        length = rng.randint(35, 160)
        if rng.random() < 0.55:
            draw.line([(x1, y1), (min(width - 40, x1 + length), y1)], fill=rng.randint(90, 165), width=1)
        else:
            draw.line([(x1, y1), (x1, min(height - 40, y1 + length))], fill=rng.randint(90, 165), width=1)
    for _ in range(10):
        x = rng.randint(50, width - 320)
        y = rng.randint(50, height - 100)
        draw.rectangle([x, y, x + rng.randint(40, 110), y + rng.randint(20, 70)], outline=rng.randint(80, 160), width=1)


def paste_lineart(base: Image.Image, pattern: Image.Image, x: int, y: int, scale: float, rotation: float) -> dict:
    new_size = (max(8, int(round(pattern.width * scale))), max(8, int(round(pattern.height * scale))))
    resized = pattern.resize(new_size, Image.Resampling.BICUBIC)
    rotated = resized.rotate(rotation, expand=True, fillcolor=255, resample=Image.Resampling.BICUBIC)
    mask = rotated.point(lambda value: 255 if value < 235 else 0)
    base.paste(rotated, (x, y), mask)
    return {"x1": x, "y1": y, "x2": x + rotated.width, "y2": y + rotated.height}


def add_scan_noise(image: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.array(image).astype(np.int16)
    noise = rng.normal(0, 5.0, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    speckles = rng.random(arr.shape) < 0.002
    arr[speckles] = rng.choice([0, 255], size=int(speckles.sum()))
    noisy = Image.fromarray(arr.astype(np.uint8), mode="L")
    return noisy.filter(ImageFilter.GaussianBlur(radius=0.25))


def generate(output_dir: str | Path = "examples", seed: int = 7) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    pattern = draw_pattern(112)
    drawing = Image.new("L", (900, 650), 255)
    draw = ImageDraw.Draw(drawing)
    add_drawing_context(draw, drawing.width, drawing.height, rng)

    placements = [
        (110, 95, 1.0, 0),
        (360, 105, 1.2, 90),
        (165, 365, 0.82, 180),
        (565, 330, 1.35, 270),
    ]
    gt = []
    for idx, (x, y, scale, rotation) in enumerate(placements, start=1):
        bbox = paste_lineart(drawing, pattern, x, y, scale, rotation)
        gt.append({"id": idx, "bbox": bbox, "scale": scale, "rotation": rotation})

    drawing = add_scan_noise(drawing, seed=seed)
    pattern_path = output / "pattern_flange.png"
    drawing_path = output / "drawing_flange.png"
    gt_path = output / "ground_truth_flange.json"
    pattern.save(pattern_path)
    drawing.save(drawing_path)
    payload = {
        "pattern": str(pattern_path.as_posix()),
        "drawing": str(drawing_path.as_posix()),
        "detections": gt,
    }
    gt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic pattern-detection examples.")
    parser.add_argument("--output-dir", default="examples")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    payload = generate(args.output_dir, args.seed)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

