from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import PatternDetector
from app.utils.io import save_image


DEFAULT_PAIRS = [
    ("drawing_01_bridge_rectifier_query.png", "1.png"),
    ("drawing_02_xor_gate_query.png", "2.png"),
    ("drawing_03_fuse_query.png", "3.png"),
    ("drawing_04_opamp_query.png", "4.png"),
    ("drawing_05_push_button_query.png", "5.png"),
    ("drawing_06_led_query.png", "6.png"),
]


def verify(drawing_dir: Path, query_dir: Path, output_dir: Path, enable_deep: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = PatternDetector(overrides={"inference": {"enable_deep": enable_deep, "log_level": "ERROR"}})
    cases = []

    for query_name, drawing_name in DEFAULT_PAIRS:
        query_path = query_dir / query_name
        drawing_path = drawing_dir / drawing_name
        result = detector.predict(query_path, drawing_path, enable_deep=enable_deep)
        rendered_path = output_dir / f"verify_{Path(drawing_name).stem}_{Path(query_name).stem}.png"
        save_image(rendered_path, result.output_image)
        cases.append(
            {
                "query_image": str(query_path.as_posix()),
                "drawing": str(drawing_path.as_posix()),
                "num_detections": len(result.detections),
                "scores": [round(det.score, 4) for det in result.detections[:12]],
                "sources": [det.source for det in result.detections[:12]],
                "rendered": str(rendered_path.as_posix()),
                "timings": result.timings,
            }
        )

    payload = {"cases": cases}
    summary_path = output_dir / "query_image_verification.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"summary": str(summary_path.as_posix()), **payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify extracted query images against their source drawings.")
    parser.add_argument("--drawing-dir", default="drawing")
    parser.add_argument("--query-dir", default="drawing/query_images")
    parser.add_argument("--output-dir", default="outputs/query_image_verification")
    parser.add_argument("--enable-deep", action="store_true")
    args = parser.parse_args()
    result = verify(Path(args.drawing_dir), Path(args.query_dir), Path(args.output_dir), args.enable_deep)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
