"""Forecasting models for AirCast Research.
"""

from typing import Any
from src.models.base import BaseForecaster
from src.models.baselines import (
    LinearRegressionForecaster,
    PersistenceForecaster,
    RidgeForecaster,
)
from src.models.registry import MODEL_REGISTRY, get_model, list_available_models
from src.models.trees import (
    HistGradientBoostingForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)


def __getattr__(name: str) -> Any:
    """Lazy-load deep learning model classes upon attribute access (PEP 562).

    Prevents importing PyTorch and allocating tensor backends during lightweight
    tree-based (XGBoost) inference or API startup.
    """
    if name == "LSTMForecaster":
        from src.models.deep_learning import LSTMForecaster
        return LSTMForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
