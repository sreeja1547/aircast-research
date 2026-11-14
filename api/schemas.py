"""Pydantic schemas for the AirCast Research API.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.features.registry import get_feature_group


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    timestamp: str = Field(..., json_schema_extra={"example": "2026-03-01T12:00:00Z"})
    available_horizons: List[int] = Field(..., json_schema_extra={"example": [1, 6, 24]})


class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    horizon_hours: int
    n_features: int
    is_loaded: bool
    artifact_file: Optional[str] = None


class ModelCatalogResponse(BaseModel):
    active_models: List[ModelInfo]
    feature_count: int
    supported_horizons: List[int]
    default_model: str


class FeatureRegistryResponse(BaseModel):
    feature_groups: Dict[str, List[str]]
    total_features_combined: int


class PredictionRequest(BaseModel):
    horizon_hours: int = Field(default=24, description="Forecasting horizon lead time: exactly 1, 6, or 24 hours.")
    model_name: Optional[str] = Field(default="recommended", description="Model architecture: 'recommended', 'xgboost', or 'persistence'.")
    features: Dict[str, float] = Field(..., description="Mapping of all 47 required features <= t.")
    compute_interval: bool = Field(default=True, description="Whether to include conformal prediction intervals.")
    nominal_coverage: float = Field(default=0.90, ge=0.5, le=0.99, description="Confidence interval coverage probability (0.50 to 0.99).")

    @field_validator("horizon_hours")
    @classmethod
    def check_supported_horizon(cls, v: int) -> int:
        if v not in {1, 6, 24}:
            raise ValueError(f"Unsupported horizon {v}. Supported horizons are {{1, 6, 24}}.")
        return v

    @field_validator("features")
    @classmethod
    def check_non_empty_features(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Features dictionary cannot be empty.")
        required = get_feature_group("combined_all")
        missing = [col for col in required if col not in v or v[col] is None]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        for col, val in v.items():
            if not np.isfinite(val):
                raise ValueError(f"Feature '{col}' has invalid non-finite value: {val}")
        return v


class PredictionResponse(BaseModel):
    model_name: str
    resolved_artifact: Optional[str] = Field(default=None, description="Exact serialized model artifact on disk.")
    horizon_hours: int
    predicted_pm25_ug_m3: float
    unit: str = "ug/m^3"
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence_level: Optional[float] = None
    interval_method: Optional[str] = Field(default=None, description="Conformal method: 'calibrated_normalized_conformal', 'calibrated_standard_conformal', or 'heuristic_fallback'")
    is_calibrated: bool = Field(default=False, description="True if interval is derived from validation empirical quantiles")
    inference_latency_ms: float


class BatchPredictionRequest(BaseModel):
    horizon_hours: int = Field(default=24, description="Forecasting horizon lead time: exactly 1, 6, or 24 hours.")
    model_name: Optional[str] = Field(default="recommended", description="Model architecture: 'recommended', 'xgboost', or 'persistence'.")
    samples: List[Dict[str, float]]
    compute_intervals: bool = Field(default=True)
    nominal_coverage: float = Field(default=0.90, ge=0.5, le=0.99, description="Confidence interval coverage probability (0.50 to 0.99).")

    @field_validator("horizon_hours")
    @classmethod
    def check_supported_horizon(cls, v: int) -> int:
        if v not in {1, 6, 24}:
            raise ValueError(f"Unsupported horizon {v}. Supported horizons are {{1, 6, 24}}.")
        return v

    @field_validator("samples")
    @classmethod
    def check_non_empty_samples(cls, v: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if not v:
            raise ValueError("Samples list cannot be empty.")
        required = get_feature_group("combined_all")
        for idx, s in enumerate(v):
            if not s:
                raise ValueError(f"Sample at index {idx} cannot be empty.")
            missing = [col for col in required if col not in s or s[col] is None]
            if missing:
                raise ValueError(f"Sample at index {idx} is missing required feature columns: {missing}")
            for col, val in s.items():
                if not np.isfinite(val):
                    raise ValueError(f"Sample at index {idx}, feature '{col}' has invalid non-finite value: {val}")
        return v


class BatchPredictionResponse(BaseModel):
    model_name: str
    resolved_artifact: Optional[str] = Field(default=None, description="Exact serialized model artifact on disk.")
    horizon_hours: int
    sample_count: int
    predictions: List[float]
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None
    confidence_level: Optional[float] = None
    interval_method: Optional[str] = None
    is_calibrated: bool = False
    total_latency_ms: float
    per_sample_latency_ms: float
