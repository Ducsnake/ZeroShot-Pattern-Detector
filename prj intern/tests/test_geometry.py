from app.utils.geometry import BBox, iou


def test_iou_partial_overlap() -> None:
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 5, 15, 15)
    assert round(iou(a, b), 4) == 0.1429


def test_bbox_clip_and_scale() -> None:
    box = BBox(-5, 2, 20, 40).clip(width=12, height=30)
    assert box.xyxy == (0.0, 2.0, 12.0, 30.0)
    assert box.scale(2).xyxy == (0.0, 4.0, 24.0, 60.0)

