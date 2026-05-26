"""Matching pipelines."""

from app.matching.classical_matcher import ClassicalFeatureMatcher
from app.matching.deep_matcher import DeepSimilarityMatcher
from app.matching.template_matcher import TemplateMatcher

__all__ = ["ClassicalFeatureMatcher", "DeepSimilarityMatcher", "TemplateMatcher"]
