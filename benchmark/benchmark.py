from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.inference import PatternDetector
from app.utils.evaluation import evaluate_detections, parse_gt_boxes
from app.utils.io import load_json


def benchmark_case(pattern: Path, drawing: Path, ground_truth: Path, config: str | None = None) -> dict:
    detector = PatternDetector(config)
    gt_payload = load_json(ground_truth)
    gt_boxes = parse_gt_boxes(gt_payload.get("detections", []))
    started = time.perf_counter()
    result = detector.predict(pattern, drawing)
    wall_time = time.perf_counter() - started
    metrics = evaluate_detections(result.detections, gt_boxes, iou_threshold=0.45)
    return {
        "pattern": str(pattern),
        "drawing": str(drawing),
        "ground_truth": str(ground_truth),
        "wall_time_seconds": wall_time,
        "pipeline_timings": result.timings,
        "num_detections": len(result.detections),
        "metrics": metrics.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU benchmark on a pattern-detection case.")
    parser.add_argument("--pattern", default="examples/pattern_flange.png")
    parser.add_argument("--drawing", default="examples/drawing_flange.png")
    parser.add_argument("--ground-truth", default="examples/ground_truth_flange.json")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="outputs/benchmark.json")
    args = parser.parse_args()

    pattern = Path(args.pattern)
    drawing = Path(args.drawing)
    gt = Path(args.ground_truth)
    if not pattern.exists() or not drawing.exists() or not gt.exists():
        from benchmark.generate_synthetic import generate

        generate("examples")

    result = benchmark_case(pattern, drawing, gt, args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

