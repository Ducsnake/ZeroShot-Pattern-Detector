"""Optional model backends."""

from app.models.dinov2 import DinoV2Model, resolve_device

__all__ = ["DinoV2Model", "resolve_device"]

