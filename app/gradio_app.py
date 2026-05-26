from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from app.inference import PatternDetector


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATTERN = ROOT / "examples" / "pattern_flange.png"
EXAMPLE_DRAWING = ROOT / "examples" / "drawing_flange.png"


def _overrides(
    score_threshold: float,
    deep_similarity_threshold: float,
    max_detections: int,
    enable_classical: bool,
    enable_template: bool,
    enable_deep: bool,
    use_dino: bool,
) -> dict[str, Any]:
    return {
        "postprocess": {
            "score_threshold": float(score_threshold),
            "max_detections": int(max_detections),
        },
        "deep": {
            "similarity_threshold": float(deep_similarity_threshold),
            "use_dinov2": bool(use_dino),
            "allow_dinov2_download": False,
        },
        "inference": {
            "enable_classical": bool(enable_classical),
            "enable_template": bool(enable_template),
            "enable_deep": bool(enable_deep),
            "skip_deep_if_candidates": True,
            "log_level": "INFO",
        },
    }


def _table_rows(payload: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for idx, det in enumerate(payload["detections"], start=1):
        box = det["bbox"]
        rows.append(
            [
                idx,
                round(float(det["score"]), 4),
                det["source"],
                round(float(box["x1"]), 1),
                round(float(box["y1"]), 1),
                round(float(box["width"]), 1),
                round(float(box["height"]), 1),
            ]
        )
    return rows


def run_inference(
    pattern_image,
    drawing_image,
    score_threshold: float,
    deep_similarity_threshold: float,
    max_detections: int,
    enable_classical: bool,
    enable_template: bool,
    enable_deep: bool,
    use_dino: bool,
):
    if pattern_image is None or drawing_image is None:
        raise gr.Error("Please upload both a pattern image and a drawing image.")

    detector = PatternDetector(
        overrides=_overrides(
            score_threshold,
            deep_similarity_threshold,
            max_detections,
            enable_classical,
            enable_template,
            enable_deep,
            use_dino,
        )
    )
    result = detector.predict(pattern_image, drawing_image)
    payload = result.to_dict()
    return result.output_image, json.dumps(payload, indent=2), _table_rows(payload)


def build_demo() -> gr.Blocks:
    css = """
    .app-wrap {max-width: 1280px; margin: 0 auto;}
    .metric-note {font-size: 0.92rem; color: #475569;}
    """
    with gr.Blocks(title="Zero-Shot Pattern Detection", css=css) as demo:
        gr.Markdown(
            "# Zero-Shot Pattern Detection",
            elem_classes=["app-wrap"],
        )
        with gr.Row(elem_classes=["app-wrap"]):
            with gr.Column(scale=1):
                pattern = gr.Image(type="pil", label="Pattern image")
                drawing = gr.Image(type="pil", label="Drawing image")
                with gr.Accordion("Inference settings", open=True):
                    score_threshold = gr.Slider(0.0, 1.0, value=0.70, step=0.01, label="Confidence threshold")
                    deep_similarity_threshold = gr.Slider(0.0, 1.0, value=0.82, step=0.01, label="Deep similarity threshold")
                    max_detections = gr.Slider(1, 200, value=100, step=1, label="Max detections")
                    enable_classical = gr.Checkbox(value=True, label="Use ORB/SIFT homography")
                    enable_template = gr.Checkbox(value=True, label="Use line-art template matching")
                    enable_deep = gr.Checkbox(value=True, label="Use deep/fallback similarity")
                    use_dino = gr.Checkbox(value=False, label="Try DINOv2 if cached")
                run = gr.Button("Run Inference", variant="primary")
                if EXAMPLE_PATTERN.exists() and EXAMPLE_DRAWING.exists():
                    gr.Examples(
                        examples=[[str(EXAMPLE_PATTERN), str(EXAMPLE_DRAWING)]],
                        inputs=[pattern, drawing],
                        label="Example images",
                    )
            with gr.Column(scale=2):
                output_image = gr.Image(type="numpy", label="Output detections")
                confidence_table = gr.Dataframe(
                    headers=["#", "Score", "Source", "x", "y", "width", "height"],
                    datatype=["number", "number", "str", "number", "number", "number", "number"],
                    label="Confidence table",
                )
                bbox_json = gr.Code(label="Bounding boxes JSON", language="json")

        run.click(
            fn=run_inference,
            inputs=[
                pattern,
                drawing,
                score_threshold,
                deep_similarity_threshold,
                max_detections,
                enable_classical,
                enable_template,
                enable_deep,
                use_dino,
            ],
            outputs=[output_image, bbox_json, confidence_table],
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
