from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


@dataclass(frozen=True)
class QueryCrop:
    source: str
    output: str
    label: str
    box_xyxy: tuple[int, int, int, int]


DEFAULT_CROPS = [
    QueryCrop("1.png", "drawing_01_bridge_rectifier_query.png", "bridge_rectifier", (380, 205, 510, 340)),
    QueryCrop("2.png", "drawing_02_xor_gate_query.png", "xor_gate", (510, 885, 565, 960)),
    QueryCrop("3.png", "drawing_03_fuse_query.png", "fuse", (125, 285, 190, 375)),
    QueryCrop("4.png", "drawing_04_opamp_query.png", "op_amp", (725, 80, 905, 215)),
    QueryCrop("5.png", "drawing_05_push_button_query.png", "push_button", (505, 520, 600, 555)),
    QueryCrop("6.png", "drawing_06_led_query.png", "led", (270, 890, 345, 960)),
]


def extract_query_images(drawing_dir: Path, output_dir: Path, contact_sheet_path: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    thumbs: list[Image.Image] = []
    labels: list[str] = []

    for spec in DEFAULT_CROPS:
        source_path = drawing_dir / spec.source
        if not source_path.exists():
            continue
        image = Image.open(source_path).convert("L")
        crop = image.crop(spec.box_xyxy)
        crop = ImageOps.autocontrast(crop)
        output_path = output_dir / spec.output
        crop.save(output_path)
        manifest.append(
            {
                "source_drawing": str(source_path.as_posix()),
                "query_image": str(output_path.as_posix()),
                "label": spec.label,
                "crop_box_xyxy": list(spec.box_xyxy),
            }
        )
        thumbs.append(crop.convert("RGB"))
        labels.append(spec.output)

    manifest_path = output_dir / "manifest.json"
    payload = {"query_images": manifest}
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if contact_sheet_path is not None and thumbs:
        contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        _save_contact_sheet(thumbs, labels, contact_sheet_path)

    return {"manifest": str(manifest_path), "created": manifest, "contact_sheet": str(contact_sheet_path) if contact_sheet_path else None}


def _save_contact_sheet(thumbs: list[Image.Image], labels: list[str], output_path: Path) -> None:
    cell_w, cell_h = 240, 180
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (thumb, label) in enumerate(zip(thumbs, labels)):
        preview = thumb.copy()
        preview.thumbnail((cell_w - 20, cell_h - 45))
        x = (idx % cols) * cell_w + 10
        y = (idx // cols) * cell_h + 10
        sheet.paste(preview, (x, y))
        draw.text((x, y + cell_h - 30), label, fill="black")
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract representative query pattern images from drawing files.")
    parser.add_argument("--drawing-dir", default="drawing")
    parser.add_argument("--output-dir", default="drawing/query_images")
    parser.add_argument("--contact-sheet", default="outputs/query_images_contact_sheet.png")
    args = parser.parse_args()

    result = extract_query_images(Path(args.drawing_dir), Path(args.output_dir), Path(args.contact_sheet))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
