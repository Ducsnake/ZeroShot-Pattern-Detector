# Evaluation Guide

The evaluation utility matches detections to ground-truth boxes greedily by IoU.

```bash
python -m benchmark.generate_synthetic --output-dir examples
python -m benchmark.benchmark
```

Output metrics:

- Precision
- Recall
- F1
- True positives
- False positives
- False negatives
- Mean IoU

For real BOM datasets, create a JSON file with:

```json
{
  "detections": [
    {"bbox": {"x1": 10, "y1": 20, "x2": 80, "y2": 100}}
  ]
}
```

Tune `deep.similarity_threshold`, `postprocess.score_threshold`, and `postprocess.nms_iou_threshold` on a validation split before production.

