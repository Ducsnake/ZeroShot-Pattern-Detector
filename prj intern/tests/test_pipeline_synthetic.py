from pathlib import Path

from app.inference import PatternDetector
from benchmark.generate_synthetic import generate


def test_pipeline_on_synthetic_example(tmp_path: Path) -> None:
    payload = generate(tmp_path)
    detector = PatternDetector(
        overrides={
            "classical": {"enabled": False},
            "deep": {
                "use_dinov2": False,
                "similarity_threshold": 0.74,
                "max_scan_side": 700,
                "max_windows": 1800,
            },
            "postprocess": {"score_threshold": 0.35, "max_detections": 20},
        }
    )
    result = detector.predict(payload["pattern"], payload["drawing"], enable_classical=False, enable_deep=True)
    assert result.detections
    assert all(det.bbox.area > 0 for det in result.detections)
    assert result.output_image.ndim == 3

