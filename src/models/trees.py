"""Tree-based ensemble forecasters for multi-horizon air quality forecasting.

Implements RandomForestForecaster, HistGradientBoostingForecaster, and
XGBoostForecaster with early stopping, feature importances, and native TreeSHAP attributions.
"""

import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb

from src.models.base import BaseForecaster


class RandomForestForecaster(BaseForecaster):
    """Random Forest Regressor Forecaster.

    Multi-threaded ensemble of de-correlated decision trees with bootstrap aggregation.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=16
        Maximum depth of individual trees.
    min_samples_split : int, default=10
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=4
        Minimum samples required at a leaf node.
    n_jobs : int, default=-1
        Number of parallel CPU jobs.
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 16,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        super().__init__(name=name, model_type="tree")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "RandomForestForecaster":
        """Fit random forest regressor."""
        t0 = time.perf_counter()
        X_valid = self._validate_inputs(X, y, is_training=True)
        y_arr = np.asarray(y, dtype=np.float64)

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.model.fit(X_valid, y_arr)

        self.is_fitted = True
        self.metadata = {
            "training_time_sec": round(time.perf_counter() - t0, 4),
            "n_samples": len(X_valid),
            "n_features": len(self.feature_names),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict PM2.5 concentrations."""
        X_valid = self._validate_inputs(X, is_training=False)
        preds = self.model.predict(X_valid)
        return np.maximum(0.0, preds)

    def get_feature_importances(self) -> pd.Series:
        """Return Gini impurity feature importances."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
            name="gini_importance",
        ).sort_values(ascending=False)


class HistGradientBoostingForecaster(BaseForecaster):
    """Histogram-based Gradient Boosted Trees Forecaster.

    Native Scikit-Learn implementation optimized for large datasets (binned histograms).

    Parameters
    ----------
    max_iter : int, default=150
        Maximum number of boosting iterations.
    learning_rate : float, default=0.08
        Step size shrinkage.
    max_depth : int, default=8
        Maximum depth of each tree.
    min_samples_leaf : int, default=20
        Minimum samples at leaf nodes.
    early_stopping : bool, default=True
        Whether to monitor validation loss to prevent overfitting.
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "hist_gradient_boosting",
        max_iter: int = 150,
        learning_rate: float = 0.08,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
        early_stopping: bool = True,
        random_state: int = 42,
    ) -> None:
        super().__init__(name=name, model_type="tree")
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.model: Optional[HistGradientBoostingRegressor] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "HistGradientBoostingForecaster":
        """Fit histogram gradient boosting model."""
        t0 = time.perf_counter()
        X_valid = self._validate_inputs(X, y, is_training=True)
        y_arr = np.asarray(y, dtype=np.float64)

        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )
        self.model.fit(X_valid, y_arr)

        self.is_fitted = True
        self.metadata = {
            "training_time_sec": round(time.perf_counter() - t0, 4),
            "n_samples": len(X_valid),
            "n_features": len(self.feature_names),
            "n_iter_": getattr(self.model, "n_iter_", self.max_iter),
            "learning_rate": self.learning_rate,
        }
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict PM2.5 concentrations."""
        X_valid = self._validate_inputs(X, is_training=False)
        preds = self.model.predict(X_valid)
        return np.maximum(0.0, preds)


class XGBoostForecaster(BaseForecaster):
    """Extreme Gradient Boosting (XGBoost) Forecaster.

    High-performance gradient boosting decision trees with histogram-based binning,
    exact second-order Taylor expansion gradients, regularization (L1/L2),
    and native C++ TreeSHAP attribution extraction.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of gradient boosted trees.
    max_depth : int, default=7
        Maximum tree depth.
    learning_rate : float, default=0.05
        Boosting learning rate (shrinkage).
    subsample : float, default=0.8
        Row subsampling ratio per tree.
    colsample_bytree : float, default=0.8
        Column subsampling ratio per tree.
    reg_alpha : float, default=0.1
        L1 regularization term on weights.
    reg_lambda : float, default=1.0
        L2 regularization term on weights.
    tree_method : str, default='hist'
        Tree construction algorithm ('hist' recommended for speed and scalability).
    early_stopping_rounds : Optional[int], default=15
        Validation rounds for early stopping if validation set is provided.
    n_jobs : int, default=-1
        Number of parallel CPU workers.
    random_state : int, default=42
        Seed for pseudo-random number generator.
    """

    def __init__(
        self,
        name: str = "xgboost",
        n_estimators: int = 200,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        early_stopping_rounds: Optional[int] = 15,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        super().__init__(name=name, model_type="tree")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model: Optional[xgb.XGBRegressor] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "XGBoostForecaster":
        """Fit XGBoost model with optional early stopping on validation split."""
        t0 = time.perf_counter()
        X_valid = self._validate_inputs(X, y, is_training=True)
        y_arr = np.asarray(y, dtype=np.float64)

        fit_params: Dict[str, Any] = {}
        eval_set = None

        if X_val is not None and y_val is not None and self.early_stopping_rounds is not None:
            self.is_fitted = True
            X_v = self._validate_inputs(X_val, is_training=False)
            self.is_fitted = False
            y_v = np.asarray(y_val, dtype=np.float64)
            eval_set = [(X_v, y_v)]

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            tree_method=self.tree_method,
            early_stopping_rounds=self.early_stopping_rounds if eval_set is not None else None,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            objective="reg:squarederror",
            eval_metric="rmse",
        )

        if eval_set is not None:
            self.model.fit(X_valid, y_arr, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_valid, y_arr, verbose=False)

        self.is_fitted = True
        best_iteration = getattr(self.model, "best_iteration", self.n_estimators)
        best_score = getattr(self.model, "best_score", None)

        self.metadata = {
            "training_time_sec": round(time.perf_counter() - t0, 4),
            "n_samples": len(X_valid),
            "n_features": len(self.feature_names),
            "n_estimators": self.n_estimators,
            "best_iteration": int(best_iteration) if best_iteration is not None else None,
            "best_val_rmse": float(best_score) if best_score is not None else None,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
        }
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict PM2.5 concentrations."""
        X_valid = self._validate_inputs(X, is_training=False)
        preds = self.model.predict(X_valid)
        return np.maximum(0.0, preds)

    def get_feature_importances(self, importance_type: str = "gain") -> pd.Series:
        """Return feature importances based on gain, weight, or cover.

        Parameters
        ----------
        importance_type : str, default='gain'
            Importance metric: 'gain', 'weight', 'cover', or 'total_gain'.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")

        booster = self.model.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)

        # Map internal feature names to column names
        importances = {}
        for i, col in enumerate(self.feature_names):
            f_key = f"f{i}"
            importances[col] = float(score_dict.get(col, score_dict.get(f_key, 0.0)))

        return pd.Series(importances, name=f"{importance_type}_importance").sort_values(
            ascending=False
        )

    def compute_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, Any]:
        """Compute exact TreeSHAP values natively using XGBoost's C++ engine.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Data to explain.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'shap_values': (n_samples, n_features) np.ndarray of feature attributions
            - 'base_value': float, model's expected baseline output
            - 'feature_names': list of feature names
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")

        X_valid = self._validate_inputs(X, is_training=False)
        if isinstance(X_valid, pd.DataFrame):
            dmat = xgb.DMatrix(X_valid.to_numpy(), feature_names=self.feature_names)
        else:
            dmat = xgb.DMatrix(X_valid, feature_names=self.feature_names)

        booster = self.model.get_booster()
        # Native TreeSHAP computation
        contribs = booster.predict(dmat, pred_contribs=True)

        # In XGBoost, last column of pred_contribs is the base expected value
        shap_vals = contribs[:, :-1]
        base_value = float(np.mean(contribs[:, -1]))

        return {
            "shap_values": shap_vals,
            "base_value": base_value,
            "feature_names": self.feature_names,
        }
