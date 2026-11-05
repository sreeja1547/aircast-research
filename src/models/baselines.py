"""Baseline forecasters for multi-horizon air quality forecasting.

Implements PersistenceForecaster (naive time-series benchmark),
LinearRegressionForecaster, and RidgeForecaster with feature scaling.
"""

import time
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.base import BaseForecaster


class PersistenceForecaster(BaseForecaster):
    """Persistence Baseline Forecaster.

    Predicts the future PM2.5 concentration at horizon t+h as the most recent
    observed PM2.5 value at time t:
        y_hat_{t+h} = y_t

    This serves as the foundational time-series benchmark. A model must beat
    persistence to demonstrate actual predictive skill.

    Parameters
    ----------
    target_col : str, default="PM2.5"
        Name of the current PM2.5 column in the input features.
    fallback_col : str, default="PM2.5_lag_1"
        Fallback column if target_col is not present in feature group.
    """

    def __init__(
        self,
        name: str = "persistence",
        target_col: str = "PM2.5",
        fallback_col: str = "PM2.5_lag_1",
    ) -> None:
        super().__init__(name=name, model_type="baseline")
        self.target_col = target_col
        self.fallback_col = fallback_col
        self.resolved_col_: Optional[Union[str, int]] = None
        self.fallback_value_: float = 0.0

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "PersistenceForecaster":
        """Record training statistics and locate the persistent predictor column."""
        t0 = time.perf_counter()
        X_valid = self._validate_inputs(X, y, is_training=True)
        y_arr = np.asarray(y, dtype=np.float64)
        self.fallback_value_ = float(np.mean(y_arr))

        if isinstance(X_valid, pd.DataFrame):
            if self.target_col in X_valid.columns:
                self.resolved_col_ = self.target_col
            elif self.fallback_col in X_valid.columns:
                self.resolved_col_ = self.fallback_col
            else:
                # Find any column containing PM2.5
                pm25_candidates = [c for c in X_valid.columns if "PM2.5" in c]
                if pm25_candidates:
                    self.resolved_col_ = pm25_candidates[0]
                else:
                    self.resolved_col_ = None
        else:
            # For raw numpy array, default to first column (index 0)
            self.resolved_col_ = 0

        self.is_fitted = True
        self.metadata = {
            "training_time_sec": round(time.perf_counter() - t0, 6),
            "n_samples": len(X_valid),
            "resolved_column": self.resolved_col_,
            "fallback_value": self.fallback_value_,
        }
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return persistent lag-0 predictions."""
        X_valid = self._validate_inputs(X, is_training=False)

        if isinstance(X_valid, pd.DataFrame):
            if self.resolved_col_ is not None and self.resolved_col_ in X_valid.columns:
                preds = X_valid[self.resolved_col_].to_numpy(dtype=np.float64, copy=True)
            else:
                preds = np.full(len(X_valid), self.fallback_value_, dtype=np.float64)
        else:
            arr = np.asarray(X_valid, dtype=np.float64)
            if self.resolved_col_ is not None and isinstance(self.resolved_col_, int):
                preds = arr[:, self.resolved_col_].copy()
            else:
                preds = np.full(len(arr), self.fallback_value_, dtype=np.float64)

        # Replace any residual NaNs with fallback value
        nan_mask = np.isnan(preds)
        if np.any(nan_mask):
            preds[nan_mask] = self.fallback_value_

        return preds


class LinearRegressionForecaster(BaseForecaster):
    """Ordinary Least Squares Linear Regression Forecaster with feature scaling.

    Parameters
    ----------
    scaler_type : str, default="robust"
        Scaler to condition features: 'robust', 'standard', or 'none'.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(
        self,
        name: str = "linear_regression",
        scaler_type: str = "robust",
        fit_intercept: bool = True,
    ) -> None:
        super().__init__(name=name, model_type="linear")
        self.scaler_type = scaler_type
        self.fit_intercept = fit_intercept
        self.scaler: Optional[Union[RobustScaler, StandardScaler]] = None
        self.model: Optional[LinearRegression] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "LinearRegressionForecaster":
        """Fit feature scaler and linear regression model."""
        t0 = time.perf_counter()
        X_valid = self._validate_inputs(X, y, is_training=True)
        y_arr = np.asarray(y, dtype=np.float64)

        if isinstance(X_valid, pd.DataFrame):
            X_mat = X_valid.to_numpy(dtype=np.float64)
        else:
            X_mat = np.asarray(X_valid, dtype=np.float64)

        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_mat)
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_mat)
        else:
            self.scaler = None
            X_scaled = X_mat

        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(X_scaled, y_arr)

        self.is_fitted = True
        self.metadata = {
            "training_time_sec": round(time.perf_counter() - t0, 4),
            "n_samples": len(X_mat),
            "n_features": X_mat.shape[1],
            "scaler_type": self.scaler_type,
            "intercept": float(self.model.intercept_),
            "max_abs_coef": float(np.max(np.abs(self.model.coef_))),
        }
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict target PM2.5 using linear combination of scaled features."""
        X_valid = self._validate_inputs(X, is_training=False)

        if isinstance(X_valid, pd.DataFrame):
            X_mat = X_valid.to_numpy(dtype=np.float64)
        else:
            X_mat = np.asarray(X_valid, dtype=np.float64)

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_mat)
        else:
            X_scaled = X_mat

        preds = self.model.predict(X_scaled)
        # Physical lower bound for pollutant concentration is zero
        return np.maximum(0.0, preds)


class RidgeForecaster(BaseForecaster):
    """L2-Regularized Ridge Regression Forecaster with feature scaling.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float.
    scaler_type : str, default="robust"
        Scaler type: 'robust', 'standard', or 'none'.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    random_state : int, default=42
        Seed for solver reproducibility.
    """

    def __init__(
        self,
        name: str = "ridge",
        alpha: float = 1.0,
        scaler_type: str = "robust",
        fit_intercept: bool = True,
        random_state: int = 42,
    ) -> None:
        super().__init__(name=name, model_type="linear")
        self.alpha = alpha
        self.scaler_type = scaler_type
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.scaler: Optional[Union[RobustScaler, StandardScaler]] = None
        self.model: Optional[Ridge] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "RidgeForecaster":
        """Fit feature scaler and ridge regression model."""
        t0 = time.perf_counter()
        X_valid = self._validate_inputs(X, y, is_training=True)
        y_arr = np.asarray(y, dtype=np.float64)

        if isinstance(X_valid, pd.DataFrame):
            X_mat = X_valid.to_numpy(dtype=np.float64)
        else:
            X_mat = np.asarray(X_valid, dtype=np.float64)

        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_mat)
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_mat)
        else:
            self.scaler = None
            X_scaled = X_mat

        self.model = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled, y_arr)

        self.is_fitted = True
        self.metadata = {
            "training_time_sec": round(time.perf_counter() - t0, 4),
            "n_samples": len(X_mat),
            "n_features": X_mat.shape[1],
            "alpha": self.alpha,
            "scaler_type": self.scaler_type,
            "intercept": float(self.model.intercept_),
            "max_abs_coef": float(np.max(np.abs(self.model.coef_))),
        }
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict PM2.5 using regularized linear regression."""
        X_valid = self._validate_inputs(X, is_training=False)

        if isinstance(X_valid, pd.DataFrame):
            X_mat = X_valid.to_numpy(dtype=np.float64)
        else:
            X_mat = np.asarray(X_valid, dtype=np.float64)

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_mat)
        else:
            X_scaled = X_mat

        preds = self.model.predict(X_scaled)
        return np.maximum(0.0, preds)
