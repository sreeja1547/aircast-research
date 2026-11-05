"""Conformal prediction for uncertainty quantification in multi-horizon air quality forecasting.

Implements inductive (split) conformal prediction with non-parametric finite-sample
coverage guarantees, adaptive interval bounding, and conditional coverage auditing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.models.base import BaseForecaster


def evaluate_conformal_coverage(
    y_true: Union[np.ndarray, pd.Series],
    lower_bound: Union[np.ndarray, pd.Series],
    upper_bound: Union[np.ndarray, pd.Series],
    nominal_coverage: float = 0.90,
    extreme_threshold: float = 150.0,
) -> Dict[str, float]:
    """Evaluate empirical interval coverage, sharpness, and conditional reliability.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth PM2.5 values.
    lower_bound : array-like of shape (n_samples,)
        Lower bound of conformal interval.
    upper_bound : array-like of shape (n_samples,)
        Upper bound of conformal interval.
    nominal_coverage : float, default=0.90
        Nominal target coverage probability (1 - alpha).
    extreme_threshold : float, default=150.0
        Threshold for severe pollution episode conditional coverage.

    Returns
    -------
    Dict[str, float]
        Dictionary with empirical coverage, width, and Winkler score.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    lb = np.asarray(lower_bound, dtype=np.float64).ravel()
    ub = np.asarray(upper_bound, dtype=np.float64).ravel()

    n = len(y_t)
    if n == 0:
        raise ValueError("Cannot evaluate conformal coverage on empty arrays.")

    # Check bounds
    covered = (y_t >= lb) & (y_t <= ub)
    empirical_coverage = float(np.mean(covered))
    interval_widths = ub - lb
    mean_width = float(np.mean(interval_widths))
    median_width = float(np.median(interval_widths))

    # Winkler Score (Interval Score):
    # S(l, u, y) = (u - l) + 2/alpha * (l - y)*I(y < l) + 2/alpha * (y - u)*I(y > u)
    alpha = 1.0 - nominal_coverage
    penalty_lower = np.maximum(0.0, lb - y_t) * (2.0 / alpha)
    penalty_upper = np.maximum(0.0, y_t - ub) * (2.0 / alpha)
    winkler_scores = interval_widths + penalty_lower + penalty_upper
    mean_winkler = float(np.mean(winkler_scores))

    # Conditional coverage on extreme pollution episodes
    extreme_mask = y_t >= extreme_threshold
    if np.sum(extreme_mask) > 0:
        extreme_coverage = float(np.mean(covered[extreme_mask]))
        extreme_width = float(np.mean(interval_widths[extreme_mask]))
    else:
        extreme_coverage = float("nan")
        extreme_width = float("nan")

    return {
        "nominal_coverage": round(nominal_coverage, 4),
        "empirical_coverage": round(empirical_coverage, 4),
        "coverage_error": round(empirical_coverage - nominal_coverage, 4),
        "mean_interval_width": round(mean_width, 4),
        "median_interval_width": round(median_width, 4),
        "mean_winkler_score": round(mean_winkler, 4),
        "extreme_coverage": round(extreme_coverage, 4) if np.isfinite(extreme_coverage) else None,
        "extreme_mean_width": round(extreme_width, 4) if np.isfinite(extreme_width) else None,
        "sample_count": n,
        "extreme_sample_count": int(np.sum(extreme_mask)),
    }


class ConformalPredictor:
    """Split Conformal Forecaster providing valid prediction intervals.

    Supports both:
    1. 'standard': Homoscedastic constant-width split conformal intervals:
       s_i = |y_i - mu_hat(x_i)|
       C(x) = [max(0, mu_hat(x) - q_hat), mu_hat(x) + q_hat]

    2. 'normalized': Heteroscedastic locally-scaled adaptive conformal intervals:
       sigma(x) = max(sigma_min, beta_0 + beta_1 * mu_hat(x))
       s_i = |y_i - mu_hat(x_i)| / sigma(x_i)
       C(x) = [max(0, mu_hat(x) - q_hat * sigma(x)), mu_hat(x) + q_hat * sigma(x)]

    Parameters
    ----------
    base_model : BaseForecaster
        Fitted underlying point forecaster.
    default_coverage : float, default=0.90
        Default nominal coverage level.
    method : str, default="standard"
        Conformal method: 'standard' or 'normalized'.
    sigma_min : float, default=5.0
        Minimum floor for difficulty score to avoid division by near-zero.
    """

    def __init__(
        self,
        base_model: BaseForecaster,
        default_coverage: float = 0.90,
        method: str = "standard",
        sigma_min: float = 5.0,
    ) -> None:
        self.base_model = base_model
        self.default_coverage = default_coverage
        self.method = method.lower().strip()
        self.sigma_min = sigma_min
        self.calibration_scores_: Optional[np.ndarray] = None
        self.is_calibrated: bool = False
        self.quantiles_: Dict[float, float] = {}
        # Dispersion parameters for normalized conformal
        self.dispersion_params_: Dict[str, float] = {"beta_0": sigma_min, "beta_1": 0.0}

    def calibrate(
        self,
        X_cal: Union[pd.DataFrame, np.ndarray],
        y_cal: Union[pd.Series, np.ndarray],
    ) -> "ConformalPredictor":
        """Compute non-conformity scores on held-out calibration partition."""
        y_c = np.asarray(y_cal, dtype=np.float64).ravel()
        point_preds = self.base_model.predict(X_cal)
        raw_residuals = np.abs(y_c - point_preds)

        if self.method == "normalized":
            # Estimate difficulty / dispersion function on calibration split:
            # sigma(y_hat) = max(sigma_min, beta_0 + beta_1 * y_hat)
            p_mean = float(np.mean(point_preds))
            r_mean = float(np.mean(raw_residuals))
            p_var = float(np.var(point_preds))
            if p_var > 1e-6:
                # OLS slope beta_1 = Cov(p, r) / Var(p)
                cov_pr = float(np.cov(point_preds, raw_residuals)[0, 1])
                beta_1 = max(0.01, cov_pr / p_var)
                beta_0 = max(self.sigma_min, r_mean - beta_1 * p_mean)
            else:
                beta_1 = 0.0
                beta_0 = max(self.sigma_min, r_mean)

            self.dispersion_params_ = {"beta_0": beta_0, "beta_1": beta_1}
            sigma_cal = np.maximum(self.sigma_min, beta_0 + beta_1 * point_preds)
            self.calibration_scores_ = raw_residuals / sigma_cal
        else:
            # Standard absolute residual non-conformity score
            self.calibration_scores_ = raw_residuals

        self.is_calibrated = True

        # Precompute quantiles for standard confidence levels
        for cov in [0.80, 0.85, 0.90, 0.95]:
            self.quantiles_[cov] = self._compute_quantile(cov)

        return self

    def _compute_quantile(self, coverage_level: float) -> float:
        """Calculate finite-sample adjusted conformal quantile."""
        if not self.is_calibrated or self.calibration_scores_ is None:
            raise RuntimeError("Conformal predictor must be calibrated before computing quantiles.")

        n = len(self.calibration_scores_)
        alpha = 1.0 - coverage_level
        # Conformal quantile level with finite sample correction (n+1)/n
        p = np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0)
        q = float(np.quantile(self.calibration_scores_, p, method="higher"))
        return q

    def predict_intervals(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        coverage_level: Optional[float] = None,
    ) -> pd.DataFrame:
        """Construct conformal prediction intervals for feature inputs."""
        cov = coverage_level if coverage_level is not None else self.default_coverage
        if cov not in self.quantiles_:
            q_val = self._compute_quantile(cov)
        else:
            q_val = self.quantiles_[cov]

        point_preds = self.base_model.predict(X)

        if self.method == "normalized":
            b0 = self.dispersion_params_.get("beta_0", self.sigma_min)
            b1 = self.dispersion_params_.get("beta_1", 0.0)
            sigma_x = np.maximum(self.sigma_min, b0 + b1 * point_preds)
            margin = q_val * sigma_x
        else:
            margin = q_val

        lower_bounds = np.maximum(0.0, point_preds - margin)
        upper_bounds = point_preds + margin

        return pd.DataFrame({
            "point_pred": point_preds,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
            "interval_width": upper_bounds - lower_bounds,
        })

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize conformal predictor state."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ConformalPredictor":
        """Deserialize conformal predictor from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Conformal artifact not found at {path}")
        return joblib.load(path)
