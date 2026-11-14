"""Unit tests for Robustness Evaluation and Stress Testing.

Verifies extreme episode stratification, monotonic noise degradation behavior,
and physical plausibility of perturbed forecasts.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_regression_metrics
from src.models import RidgeForecaster


def test_extreme_event_stratification():
    """Verify performance partitioning on extreme haze episodes."""
    y_true = np.array([25.0, 50.0, 155.0, 260.0])
    y_pred = np.array([28.0, 48.0, 160.0, 245.0])

    metrics = calculate_regression_metrics(y_true, y_pred, extreme_threshold=150.0)
    assert metrics["extreme_count"] == 2
    assert metrics["normal_count"] == 2
    assert metrics["extreme_mae"] == pytest.approx(10.0, abs=1e-3)
    assert metrics["normal_mae"] == pytest.approx(2.5, abs=1e-3)


def test_meaningful_noise_degradation_behavior():
    """Verify that increasing sensor noise monotonically degrades predictive skill (higher MAE, lower R2)."""
    rng = np.random.default_rng(42)
    n_train, n_test = 200, 100

    # Synthetic atmospheric physical relationship: PM2.5 = 0.75 * lag_1 - 3.0 * WSPM + noise
    def generate_data(n: int):
        pm25_lag = rng.uniform(20.0, 180.0, n)
        wspm = rng.uniform(0.5, 6.0, n)
        y = 0.75 * pm25_lag - 3.0 * wspm + rng.normal(0, 5.0, n)
        y = np.maximum(0.0, y)
        X = pd.DataFrame({"PM2.5_lag_1": pm25_lag, "WSPM": wspm})
        return X, pd.Series(y)

    X_train, y_train = generate_data(n_train)
    X_test, y_test = generate_data(n_test)

    forecaster = RidgeForecaster(name="ridge_robustness_test", alpha=1.0)
    forecaster.fit(X_train, y_train)

    # 1. Clean test evaluation
    clean_preds = forecaster.predict(X_test)
    clean_metrics = calculate_regression_metrics(y_test.values, clean_preds)

    # 2. Mild noise perturbation (sigma = 15.0)
    X_test_mild = X_test.copy()
    X_test_mild["PM2.5_lag_1"] += rng.normal(0, 15.0, n_test)
    mild_preds = forecaster.predict(X_test_mild)
    mild_metrics = calculate_regression_metrics(y_test.values, mild_preds)

    # 3. Severe noise perturbation (sigma = 45.0)
    X_test_severe = X_test.copy()
    X_test_severe["PM2.5_lag_1"] += rng.normal(0, 45.0, n_test)
    severe_preds = forecaster.predict(X_test_severe)
    severe_metrics = calculate_regression_metrics(y_test.values, severe_preds)

    # Verify strict monotonic degradation: Clean < Mild < Severe MAE
    assert clean_metrics["mae"] < mild_metrics["mae"], (
        f"Expected clean MAE ({clean_metrics['mae']:.2f}) < mild MAE ({mild_metrics['mae']:.2f})"
    )
    assert mild_metrics["mae"] < severe_metrics["mae"], (
        f"Expected mild MAE ({mild_metrics['mae']:.2f}) < severe MAE ({severe_metrics['mae']:.2f})"
    )

    # Verify monotonic degradation of R2: Clean > Mild > Severe R2
    assert clean_metrics["r2"] > mild_metrics["r2"], (
        f"Expected clean R2 ({clean_metrics['r2']:.4f}) > mild R2 ({mild_metrics['r2']:.4f})"
    )
    assert mild_metrics["r2"] > severe_metrics["r2"], (
        f"Expected mild R2 ({mild_metrics['r2']:.4f}) > severe R2 ({severe_metrics['r2']:.4f})"
    )
