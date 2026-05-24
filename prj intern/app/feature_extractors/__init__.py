"""Feature extraction backends."""

from app.feature_extractors.classical import FeatureSet, extract_feature_sets
from app.feature_extractors.deep import BaseEmbeddingExtractor, build_embedding_extractor

__all__ = ["BaseEmbeddingExtractor", "FeatureSet", "build_embedding_extractor", "extract_feature_sets"]

