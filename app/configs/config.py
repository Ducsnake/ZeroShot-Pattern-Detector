from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


T = TypeVar("T")


@dataclass
class PreprocessConfig:
    max_image_side: int = 1800
    denoise: bool = True
    denoise_strength: int = 7
    adaptive_threshold: bool = True
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    morphology: bool = True
    morphology_kernel: int = 2
    edge_enhance: bool = True
    invert_if_dark_foreground: bool = True


@dataclass
class ClassicalConfig:
    enabled: bool = True
    use_sift: bool = True
    use_orb: bool = True
    orb_features: int = 5000
    sift_features: int = 3500
    ratio_test: float = 0.75
    min_matches: int = 10
    min_inliers: int = 12
    ransac_reproj_threshold: float = 4.0
    max_instances_per_variant: int = 15
    rotations: list[float] = field(default_factory=lambda: [0, 90, 180, 270])
    scales: list[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])


@dataclass
class TemplateConfig:
    enabled: bool = True
    threshold: float = 0.48
    min_foreground_f1: float = 0.32
    min_combined_score: float = 0.70
    rotations: list[float] = field(default_factory=lambda: [0, 90, 180, 270])
    scales: list[float] = field(default_factory=lambda: [0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.0])
    max_candidates_per_variant: int = 8
    local_max_kernel: int = 9
    min_template_size: int = 16


@dataclass
class DeepConfig:
    enabled: bool = True
    use_dinov2: bool = True
    allow_dinov2_download: bool = False
    model_name: str = "dinov2_vits14"
    device: str = "auto"
    input_size: int = 224
    batch_size: int = 16
    max_scan_side: int = 1400
    tile_size: int = 640
    tile_overlap_ratio: float = 0.35
    step_ratio: float = 0.25
    min_window: int = 24
    max_windows: int = 2500
    similarity_threshold: float = 0.82
    scales: list[float] = field(default_factory=lambda: [0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5])
    rotations: list[float] = field(default_factory=lambda: [0, 90, 180, 270])
    heatmap_smoothing_sigma: float = 1.2


@dataclass
class PostprocessConfig:
    score_threshold: float = 0.70
    nms_iou_threshold: float = 0.25
    duplicate_iou_threshold: float = 0.65
    max_detections: int = 100
    min_box_area: int = 64


@dataclass
class InferenceConfig:
    enable_classical: bool = True
    enable_template: bool = True
    enable_deep: bool = True
    skip_deep_if_candidates: bool = True
    deep_skip_score_threshold: float = 0.78
    timeout_seconds: int = 60
    cache_embeddings: bool = True
    log_level: str = "INFO"


@dataclass
class AppConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")


def _update_dataclass(instance: T, values: Mapping[str, Any]) -> T:
    valid_fields = {item.name: item for item in fields(instance)}
    for key, value in values.items():
        if key not in valid_fields:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(config_path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """Load YAML config and optional nested overrides into typed dataclasses."""

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = AppConfig()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"Config file must contain a mapping: {path}")
        _update_dataclass(config, data)

    if overrides:
        _update_dataclass(config, overrides)

    config.preprocess.adaptive_block_size = max(3, int(config.preprocess.adaptive_block_size) | 1)
    config.preprocess.morphology_kernel = max(1, int(config.preprocess.morphology_kernel))
    config.template.threshold = min(max(float(config.template.threshold), 0.0), 1.0)
    config.template.min_foreground_f1 = min(max(float(config.template.min_foreground_f1), 0.0), 1.0)
    config.template.min_combined_score = min(max(float(config.template.min_combined_score), 0.0), 1.0)
    config.template.local_max_kernel = max(3, int(config.template.local_max_kernel) | 1)
    config.deep.step_ratio = min(max(float(config.deep.step_ratio), 0.05), 1.0)
    config.deep.tile_overlap_ratio = min(max(float(config.deep.tile_overlap_ratio), 0.0), 0.9)
    config.inference.deep_skip_score_threshold = min(max(float(config.inference.deep_skip_score_threshold), 0.0), 1.0)
    config.postprocess.score_threshold = min(max(float(config.postprocess.score_threshold), 0.0), 1.0)
    return config
