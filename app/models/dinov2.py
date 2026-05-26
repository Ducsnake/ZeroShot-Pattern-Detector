from __future__ import annotations

import os
from pathlib import Path

import torch

from app.utils.exceptions import ModelLoadError
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return requested


def _hub_repo_candidates() -> list[Path]:
    hub_dir = Path(torch.hub.get_dir())
    return [
        hub_dir / "facebookresearch_dinov2_main",
        hub_dir / "facebookresearch_dinov2_master",
    ]


def dinov2_cache_available(model_name: str = "dinov2_vits14") -> bool:
    hub_dir = Path(torch.hub.get_dir())
    repo_available = any(path.exists() for path in _hub_repo_candidates())
    checkpoints = list((hub_dir / "checkpoints").glob(f"{model_name}*.pth"))
    return repo_available and bool(checkpoints)


class DinoV2Model:
    """Lazy DINOv2 loader.

    The project can run without downloading DINOv2. Set allow_download=True or
    ZD_ALLOW_DINO_DOWNLOAD=1 when the runtime may access torch hub.
    """

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "auto", allow_download: bool = False) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.allow_download = allow_download or os.getenv("ZD_ALLOW_DINO_DOWNLOAD", "0") == "1"
        self._model: torch.nn.Module | None = None

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> torch.nn.Module:
        if not self.allow_download and not dinov2_cache_available(self.model_name):
            raise ModelLoadError(
                "DINOv2 cache was not found. Set ZD_ALLOW_DINO_DOWNLOAD=1 or "
                "deep.allow_dinov2_download=true to let torch.hub download weights."
            )
        try:
            model = torch.hub.load("facebookresearch/dinov2", self.model_name, trust_repo=True)
        except Exception as exc:  # pragma: no cover - network/cache dependent
            raise ModelLoadError(f"Could not load DINOv2 model {self.model_name}: {exc}") from exc
        model.eval().to(self.device)
        for param in model.parameters():
            param.requires_grad_(False)
        LOGGER.info("Loaded DINOv2 model %s on %s", self.model_name, self.device)
        return model

    @torch.inference_mode()
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device, non_blocking=True)
        output = self.model(batch)
        if isinstance(output, dict):
            if "x_norm_clstoken" in output:
                output = output["x_norm_clstoken"]
            elif "x_prenorm" in output:
                output = output["x_prenorm"][:, 0]
            else:
                first_value = next(iter(output.values()))
                output = first_value[:, 0] if first_value.ndim == 3 else first_value
        if output.ndim == 3:
            output = output[:, 0]
        return torch.nn.functional.normalize(output.float(), dim=1).cpu()
