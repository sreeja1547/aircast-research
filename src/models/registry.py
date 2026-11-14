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
from src.models.trees import (
    HistGradientBoostingForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)


class _LazyModelRegistry(dict):
    """Dictionary that lazily loads deep learning forecasters upon access.

    Avoids importing PyTorch and allocating tensor backends during lightweight
    tree-based inference or API service startup.
    """

    def __getitem__(self, key: str) -> Type[BaseForecaster]:
        clean_key = str(key).lower().strip()
        if clean_key == "lstm" and not super().__contains__("lstm"):
            from src.models.deep_learning import LSTMForecaster
            super().__setitem__("lstm", LSTMForecaster)
        return super().__getitem__(clean_key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and key.lower().strip() == "lstm":
            return True
        return super().__contains__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        all_keys = list(super().keys())
        if "lstm" not in all_keys:
            all_keys.append("lstm")
        return all_keys

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())


MODEL_REGISTRY: Dict[str, Type[BaseForecaster]] = _LazyModelRegistry({
    "persistence": PersistenceForecaster,
    "linear_regression": LinearRegressionForecaster,
    "ridge": RidgeForecaster,
    "random_forest": RandomForestForecaster,
    "hist_gradient_boosting": HistGradientBoostingForecaster,
    "xgboost": XGBoostForecaster,
})


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
