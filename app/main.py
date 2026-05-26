from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.inference import PatternDetector


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "postprocess": {
            "score_threshold": args.score_threshold,
            "max_detections": args.max_detections,
        },
        "deep": {
            "similarity_threshold": args.deep_similarity_threshold,
            "use_dinov2": args.use_dino,
            "allow_dinov2_download": args.allow_dino_download,
        },
        "inference": {
            "enable_classical": not args.no_classical,
            "enable_template": not args.no_template,
            "enable_deep": not args.no_deep,
            "log_level": args.log_level,
        },
    }
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot pattern detection for technical drawings.")
    parser.add_argument("--pattern", required=True, help="Path to query pattern image.")
    parser.add_argument("--drawing", required=True, help="Path to drawing image.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for rendered image and JSON.")
    parser.add_argument("--prefix", default="result", help="Output filename prefix.")
    parser.add_argument("--score-threshold", type=float, default=0.70, help="Post-processing confidence threshold.")
    parser.add_argument("--deep-similarity-threshold", type=float, default=0.82, help="Raw deep cosine threshold.")
    parser.add_argument("--max-detections", type=int, default=100, help="Maximum detections after NMS.")
    parser.add_argument("--no-classical", action="store_true", help="Disable ORB/SIFT homography matching.")
    parser.add_argument("--no-template", action="store_true", help="Disable line-art template NCC matching.")
    parser.add_argument("--no-deep", action="store_true", help="Disable deep/fallback similarity matching.")
    parser.add_argument("--use-dino", action="store_true", help="Try DINOv2 embeddings before line-art fallback.")
    parser.add_argument("--allow-dino-download", action="store_true", help="Allow torch.hub to download DINOv2 weights.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = PatternDetector(args.config, overrides=_build_overrides(args))
    result = detector.predict(args.pattern, args.drawing)
    image_path, json_path = detector.export(result, output_dir=Path(args.output_dir), prefix=args.prefix)
    print(json.dumps({**result.to_dict(), "output_image": str(image_path), "output_json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
