from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.models.dinov2 import DinoV2Model


class DinoV2OnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                out = out["x_norm_clstoken"]
            elif "x_prenorm" in out:
                out = out["x_prenorm"][:, 0]
            else:
                first_value = next(iter(out.values()))
                out = first_value[:, 0] if first_value.ndim == 3 else first_value
        if out.ndim == 3:
            out = out[:, 0]
        return torch.nn.functional.normalize(out.float(), dim=1)


def export_dinov2_onnx(
    output_path: str | Path = "outputs/dinov2_vits14.onnx",
    model_name: str = "dinov2_vits14",
    input_size: int = 224,
    allow_download: bool = False,
) -> Path:
    dino = DinoV2Model(model_name=model_name, device="cpu", allow_download=allow_download)
    wrapper = DinoV2OnnxWrapper(dino.model).eval()
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cached DINOv2 model to ONNX.")
    parser.add_argument("--output", default="outputs/dinov2_vits14.onnx")
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    print(export_dinov2_onnx(args.output, args.model_name, args.input_size, args.allow_download))


if __name__ == "__main__":
    main()

