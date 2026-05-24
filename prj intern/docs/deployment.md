# Deployment Guide

## Local Gradio

```bash
pip install -r requirements.txt
python -m benchmark.generate_synthetic --output-dir examples
python -m app.gradio_app
```

## Docker

```bash
docker build -t zero-shot-pattern-detector .
docker run --rm -p 7860:7860 zero-shot-pattern-detector
```

## HuggingFace Spaces

Use SDK `Gradio`. The README metadata points the Space to `app/gradio_app.py`.

For deterministic offline CPU deployment, keep DINOv2 disabled. To use DINOv2, pre-cache the model in the Space persistent storage or set `ZD_ALLOW_DINO_DOWNLOAD=1`.

## Production Runtime Recommendations

- Run behind a request queue for large drawings.
- Keep `preprocess.max_image_side`, `deep.max_scan_side`, and `deep.max_windows` bounded.
- Persist output JSON for audits.
- Capture per-stage timings from `InferenceResult.timings`.
- Version the config file with each deployment.

