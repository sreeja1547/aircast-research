"""Model registry and factory for AirCast forecasting architectures.

Provides centralized model catalog, instantiators, and configuration bindings
across baselines, linear models, tree ensembles, and deep recurrent neural networks.
"""

from typing import Any, Dict, List, Type
from src.models.base import BaseForecaster
from src.models.baselines import (
    LinearRegressionForecaster,
    PersistenceForecaster,
    RidgeForecaster,
)
from src.models.deep_learning import LSTMForecaster
from src.models.trees import (
    HistGradientBoostingForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)

MODEL_REGISTRY: Dict[str, Type[BaseForecaster]] = {
    "persistence": PersistenceForecaster,
    "linear_regression": LinearRegressionForecaster,
    "ridge": RidgeForecaster,
    "random_forest": RandomForestForecaster,
    "hist_gradient_boosting": HistGradientBoostingForecaster,
    "xgboost": XGBoostForecaster,
    "lstm": LSTMForecaster,
}


def list_available_models() -> List[str]:
    """Return list of all registered model identifiers."""
    return list(MODEL_REGISTRY.keys())


def get_model(name: str, **kwargs: Any) -> BaseForecaster:
    """Instantiate a forecaster from the registry by its identifier.

    Parameters
    ----------
    name : str
        Canonical identifier: 'persistence', 'linear_regression', 'ridge',
        'random_forest', 'hist_gradient_boosting', 'xgboost', or 'lstm'.
    **kwargs : Any
        Keyword arguments passed directly to the model's constructor.

    Returns
    -------
    BaseForecaster
        Configured forecaster instance.
    """
    canonical_name = name.lower().strip()
    if canonical_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available registered models are: {list_available_models()}"
        )

    model_cls = MODEL_REGISTRY[canonical_name]
    return model_cls(**kwargs)
