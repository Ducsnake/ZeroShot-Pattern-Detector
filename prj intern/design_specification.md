# Design Specification: Zero-Shot Pattern Detection for Technical BOM Drawings

## 1. Problem Analysis

The task is to find every occurrence of a query pattern inside a technical drawing. The query can change per request, so a supervised detector trained on fixed classes is not suitable. The detector must work with thin strokes, binary/grayscale drawings, scan noise, scale changes, rotation changes, and multiple simultaneous objects.

The chosen design is a hybrid zero-shot pipeline:

1. Classical local features detect geometrically verifiable instances when the drawing has enough keypoints.
2. Deep or deterministic line-art embeddings provide dense similarity search when keypoints are sparse.
3. Post-processing merges duplicate evidence and produces stable final boxes.

This is deliberately modular. A production deployment can tune, replace, or disable each stage without changing the public inference API.

## 2. Why Hybrid Instead of a Single Method

### Classical-only limitations

SIFT and ORB are strong for local geometric matching, but they can fail on minimal line drawings with repeated straight segments, tiny symbols, or aggressive binarization. They also tend to return one dominant homography unless iterative instance extraction is implemented.

### Deep-only limitations

DINOv2 embeddings are strong generic visual descriptors, but full sliding-window search can be expensive on CPU and may need the model weights cached. Deep embeddings also do not provide geometric verification by themselves.

### Hybrid benefits

The classical path gives high-precision homography detections. The deep/fallback path improves recall for sparse or noisy symbols. NMS and duplicate merging combine both evidence streams.

## 3. End-to-End Pipeline

### Inputs

- Query pattern image: PNG/JPEG/TIFF/BMP or any OpenCV/Pillow-readable image.
- Drawing image: grayscale, binary, or RGB engineering drawing.

### Outputs

- List of detections:
  - `bbox`: `x1`, `y1`, `x2`, `y2`, `width`, `height`
  - `score`: confidence in `[0, 1]`
  - `source`: matcher path, for example `classical-sift` or `deep-lineart-hog`
  - `metadata`: rotation, scale, inliers, similarity, and merge details
- Rendered image with bounding boxes.
- JSON export.

## 4. Module Details

### 4.1 Preprocessing: `app/preprocess.py`

The preprocessing stage normalizes the two images before matching:

- Convert to grayscale.
- Normalize dark-background scans to white-background line art.
- Denoise with fast non-local means.
- Enhance edges with CLAHE and unsharp masking.
- Adaptive threshold for uneven scan illumination.
- Morphology close/open with a small kernel.
- Resize large drawings to keep CPU latency bounded.
- Crop foreground for compact query descriptors.
- Rotate query variants while keeping full bounds.

The output remains an 8-bit image so OpenCV, scikit-image, and PyTorch paths can share it.

### 4.2 Classical Matching

Files:

- `app/feature_extractors/classical.py`
- `app/matching/classical_matcher.py`

Steps:

1. Generate scaled and rotated query variants.
2. Extract SIFT and ORB features where available.
3. Extract drawing features once per request.
4. Match descriptors with BFMatcher.
5. Apply Lowe ratio filtering.
6. Estimate homography with RANSAC.
7. Convert projected query corners into an axis-aligned bbox.
8. Remove inlier matches and repeat RANSAC to find additional instances.

Confidence uses:

- Inlier ratio.
- Number of inliers.
- Median descriptor distance.
- Basic geometric sanity checks.

This path is high precision and naturally scale/rotation tolerant through local features and query variants.

### 4.3 Deep Feature Similarity

Files:

- `app/feature_extractors/deep.py`
- `app/models/dinov2.py`
- `app/matching/deep_matcher.py`

The deep path supports two embedding backends:

1. `DinoV2EmbeddingExtractor`: uses `torch.hub.load("facebookresearch/dinov2", model_name)` when weights are cached or download is explicitly allowed.
2. `LineArtEmbeddingExtractor`: deterministic CPU fallback using resized foreground pixels, HOG, Canny edge summaries, and horizontal/vertical projection profiles.

The matcher:

- Creates rotated query embeddings.
- Scans the drawing with multi-scale tiled windows.
- Batches patch embedding extraction.
- Computes cosine similarity.
- Keeps windows above `deep.similarity_threshold`.
- Accumulates a similarity heatmap for debugging and visualization.

This gives a zero-shot dense search mechanism that still runs without model download.

### 4.4 Post-Processing

Files:

- `app/postprocess/calibration.py`
- `app/postprocess/nms.py`

Steps:

1. Drop low-score or tiny-area boxes.
2. Calibrate scores by source type.
3. Clip boxes to image bounds.
4. Merge high-IoU duplicates from different matchers.
5. Apply NMS with configurable IoU.
6. Cap final detections.

Merging retains member metadata so debugging can show whether a result came from SIFT, ORB, embedding similarity, or a consensus.

### 4.5 Visualization

File: `app/visualization/draw.py`

The visual layer renders:

- Axis-aligned boxes.
- Numeric confidence labels.
- Optional heatmap overlay.
- JSON export.

The rendered image uses RGB arrays to work directly with Gradio and OpenCV save utilities.

### 4.6 Inference API

File: `app/inference.py`

Public class: `PatternDetector`.

Core methods:

- `predict(pattern_image, drawing_image) -> InferenceResult`
- `predict_async(...) -> InferenceResult`
- `batch_predict(patterns, drawing_image) -> list[InferenceResult]`
- `export(result, output_dir, prefix) -> image_path, json_path`

The API accepts paths, Pillow images, or NumPy arrays. It returns structured detections and rendered output.

## 5. Configuration System

Default config: `app/configs/default.yaml`.

The loader in `app/configs/config.py` maps YAML to typed dataclasses and accepts nested runtime overrides. This keeps deployment behavior explicit and avoids hardcoded thresholds.

Primary config groups:

- `preprocess`
- `classical`
- `deep`
- `postprocess`
- `inference`

## 6. CPU and Memory Optimization

Implemented controls:

- Resize high-resolution drawings with `preprocess.max_image_side`.
- Separate scan-side cap for deep matching through `deep.max_scan_side`.
- Overlapping tiled scan through `deep.tile_size` and `deep.tile_overlap_ratio`.
- Sliding-window cap via `deep.max_windows`.
- Batched embedding extraction with `deep.batch_size`.
- Lazy DINOv2 loading.
- Offline fallback extractor.
- Inlier removal in classical matcher to avoid unnecessary repeated full matching.

Expected CPU time depends on drawing size, window count, and DINO availability. The default config is set for practical sub-minute requests on moderate-resolution drawings.

## 7. Handling Scale and Rotation

Scale is handled by:

- Query image pyramids in classical matching.
- Multi-scale sliding windows in deep matching.
- Local feature scale invariance from SIFT/ORB.

Rotation is handled by:

- Query variants at configured angles.
- SIFT/ORB orientation assignment.
- Window matching against rotated query embeddings.

For arbitrary rotations, add angles such as `-45, -30, -15, 15, 30, 45` to the config. Runtime increases roughly linearly with the number of angles.

## 8. Evaluation

Files:

- `app/utils/evaluation.py`
- `benchmark/benchmark.py`
- `benchmark/generate_synthetic.py`

Metrics:

- Precision
- Recall
- F1
- True positives
- False positives
- False negatives
- Mean IoU

The synthetic generator creates repeatable line-art data with noise, scale variation, rotation variation, and ground truth boxes.

## 9. Trade-Offs

### Accuracy versus latency

Smaller sliding-window steps and more rotation angles improve recall but increase CPU cost. The defaults favor production latency.

### DINOv2 versus fallback descriptors

DINOv2 gives better semantic robustness but needs weights and more compute. The fallback extractor is deterministic, lightweight, and strong for binary technical symbols, but it is less robust to severe drawing style shifts.

### Axis-aligned boxes versus polygons

Homography produces projected quadrilaterals, but the output contract uses bounding boxes. Axis-aligned boxes simplify downstream BOM workflows and UI display. Polygon export can be added from the same projected corners if needed.

### Classical verification versus dense search

Classical matching is precise but can miss low-texture patterns. Dense search is more exhaustive but can produce nearby duplicate candidates. The postprocessor reconciles both.

## 10. Current Limitations

- DINOv2 download is disabled by default to keep offline deployments predictable.
- Dense scanning can miss very small objects if `deep.min_window` is too high.
- Heavy arbitrary rotation search requires adding more configured angles.
- Complex overlapping symbols may need contour proposals or segmentation before matching.
- Confidence calibration is heuristic and should be calibrated on production validation data.

## 11. Production Deployment Notes

Recommended deployment settings:

- Pre-generate example assets during image build.
- Cache DINOv2 weights in the container or persistent volume if using DINO.
- Keep `deep.max_windows` bounded for latency SLOs.
- Log timings per stage and alert on slow requests.
- Store JSON outputs for auditability.
- Add request-level tracing around `PatternDetector.predict`.
- Use a queue for large batch workloads.

## 12. Future Improvements

- Contour or connected-component proposal generation to reduce sliding-window count.
- ONNX Runtime backend for DINOv2.
- FAISS or ANN index for large batches of patterns.
- Learned calibration from labeled validation drawings.
- Rotated bbox output and polygon output.
- Active error mining from false positives and false negatives.
- Domain-specific symbol normalization for BOM tables and CAD exports.
