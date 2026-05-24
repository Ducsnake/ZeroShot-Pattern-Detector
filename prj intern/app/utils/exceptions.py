class PatternDetectionError(RuntimeError):
    """Base exception for pattern detection failures."""


class ImageLoadError(PatternDetectionError):
    """Raised when an input image cannot be decoded."""


class ModelLoadError(PatternDetectionError):
    """Raised when an optional model cannot be loaded."""

