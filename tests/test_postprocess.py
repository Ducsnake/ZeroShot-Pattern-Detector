from app.configs.config import PostprocessConfig
from app.postprocess import PostProcessor
from app.utils.geometry import BBox, Detection


def test_postprocess_merges_duplicate_boxes() -> None:
    processor = PostProcessor(PostprocessConfig(score_threshold=0.1, duplicate_iou_threshold=0.5, nms_iou_threshold=0.3))
    detections = [
        Detection(BBox(10, 10, 50, 50), 0.8, "deep-lineart"),
        Detection(BBox(12, 12, 52, 52), 0.7, "classical-orb"),
        Detection(BBox(100, 100, 150, 150), 0.6, "deep-lineart"),
    ]
    result = processor.run(detections, image_shape=(200, 200))
    assert len(result) == 2
    assert result[0].score >= 0.8

