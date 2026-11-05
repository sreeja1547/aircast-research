"""Pydantic schemas for the AirCast Research API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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


class ModelCatalogResponse(BaseModel):
    active_models: List[ModelInfo]
    feature_count: int
    supported_horizons: List[int]
    default_model: str


class PredictionRequest(BaseModel):
    horizon_hours: int = Field(default=24, ge=1, le=24, description="Forecasting horizon (1, 6, or 24 hours)")
    model_name: Optional[str] = Field(default="xgboost", description="Model architecture: 'xgboost', 'hist_gradient_boosting', 'ridge', or 'persistence'")
    features: Dict[str, float] = Field(..., description="Mapping of feature names to observed values <= t")
    compute_interval: bool = Field(default=True, description="Whether to include conformal prediction intervals")
    nominal_coverage: float = Field(default=0.90, ge=0.5, le=0.99, description="Confidence interval coverage probability")


class PredictionResponse(BaseModel):
    model_name: str
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
    horizon_hours: int = Field(default=24, ge=1, le=24)
    model_name: Optional[str] = Field(default="xgboost")
    samples: List[Dict[str, float]]
    compute_intervals: bool = Field(default=True)
    nominal_coverage: float = Field(default=0.90)


class BatchPredictionResponse(BaseModel):
    model_name: str
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


class FeatureRegistryResponse(BaseModel):
    feature_groups: Dict[str, List[str]]
    total_features_combined: int
