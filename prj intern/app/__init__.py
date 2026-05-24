"""Zero-shot pattern detection package."""

__all__ = ["PatternDetector"]


def __getattr__(name: str):
    if name == "PatternDetector":
        from app.inference import PatternDetector

        return PatternDetector
    raise AttributeError(name)
