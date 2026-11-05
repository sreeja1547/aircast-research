"""Preprocessing module for AirCast Research.

Includes station-aware causal imputation, purged temporal splitting,
and strictly leak-free transformations.
"""

from .cleaner import StationAwareCausalImputer
from .splitter import TemporalPurgedSplitter, StationHoldoutSplitter
from .transformers import CyclicalTransformer, WindVectorTransformer

__all__ = [
    "StationAwareCausalImputer",
    "TemporalPurgedSplitter",
    "StationHoldoutSplitter",
    "CyclicalTransformer",
    "WindVectorTransformer",
]
