"""Unit tests for Robustness Evaluation and Stress Testing.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_regression_metrics
from src.models import PersistenceForecaster


def test_extreme_event_stratification():
    """Verify performance partitioning on extreme haze episodes."""
    y_true = np.array([25.0, 50.0, 155.0, 260.0])
    y_pred = np.array([28.0, 48.0, 160.0, 245.0])

    metrics = calculate_regression_metrics(y_true, y_pred, extreme_threshold=150.0)
    assert metrics["extreme_count"] == 2
    assert metrics["normal_count"] == 2
    assert metrics["extreme_mae"] == pytest.approx(10.0, abs=1e-3)
    assert metrics["normal_mae"] == pytest.approx(2.5, abs=1e-3)


def test_noise_degradation_behavior():
    """Verify that adding measurement noise does not crash model predict."""
    np.random.seed(42)
    df = pd.DataFrame({
        "PM2.5": np.random.uniform(20.0, 100.0, 50),
        "WSPM": np.random.uniform(0.5, 5.0, 50),
    })
    model = PersistenceForecaster(target_col="PM2.5")
    model.fit(df, df["PM2.5"])

    # Inject Gaussian noise
    noisy_df = df.copy()
    noisy_df["PM2.5"] += np.random.normal(0, 10.0, len(df))
    preds = model.predict(noisy_df)
    assert len(preds) == len(df)
    assert not np.isnan(preds).any()
