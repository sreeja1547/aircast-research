"""Forecasting models for AirCast Research.
"""

from src.models.base import BaseForecaster
from src.models.baselines import (
    LinearRegressionForecaster,
    PersistenceForecaster,
    RidgeForecaster,
)
from src.models.deep_learning import LSTMForecaster
from src.models.registry import MODEL_REGISTRY, get_model, list_available_models
from src.models.trees import (
    HistGradientBoostingForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)

__all__ = [
    "BaseForecaster",
    "PersistenceForecaster",
    "LinearRegressionForecaster",
    "RidgeForecaster",
    "RandomForestForecaster",
    "HistGradientBoostingForecaster",
    "XGBoostForecaster",
    "LSTMForecaster",
    "MODEL_REGISTRY",
    "get_model",
    "list_available_models",
]
