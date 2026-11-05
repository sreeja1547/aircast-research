"""Unit tests for evaluation metrics and latency benchmarking.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    benchmark_latency,
    calculate_regression_metrics,
    evaluate_by_subgroups,
)


def test_perfect_prediction_metrics():
    """Verify metrics on identical ground truth and predictions."""
    y = np.array([25.0, 50.0, 100.0, 200.0, 300.0])
    metrics = calculate_regression_metrics(y, y)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["mape"] == 0.0
    assert metrics["smape"] == 0.0
    assert metrics["pearson_r"] == 1.0
    assert metrics["sample_count"] == 5
    assert metrics["extreme_count"] == 2  # 200.0 and 300.0
    assert metrics["normal_count"] == 3


def test_known_error_metrics():
    """Verify metrics calculation against hand-computed values."""
    y_true = np.array([10.0, 20.0, 30.0, 160.0])
    y_pred = np.array([12.0, 18.0, 33.0, 150.0])

    # Absolute errors: [2, 2, 3, 10], Mean = 4.25
    # Squared errors: [4, 4, 9, 100], Mean = 29.25, sqrt = 5.4083
    metrics = calculate_regression_metrics(y_true, y_pred, extreme_threshold=150.0)

    assert metrics["mae"] == pytest.approx(4.25, abs=1e-3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(29.25), abs=1e-3)
    assert metrics["sample_count"] == 4
    assert metrics["extreme_count"] == 1
    assert metrics["extreme_mae"] == 10.0
    assert metrics["normal_count"] == 3
    assert metrics["normal_mae"] == pytest.approx(7.0 / 3.0, abs=1e-3)


def test_invalid_inputs_raise_errors():
    """Verify error handling on length mismatch and empty arrays."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_regression_metrics([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="Cannot calculate metrics on empty arrays"):
        calculate_regression_metrics([], [])


def test_evaluate_by_subgroups():
    """Verify grouped evaluation across stations."""
    y_true = np.array([20.0, 30.0, 180.0, 220.0])
    y_pred = np.array([22.0, 28.0, 175.0, 230.0])
    groups = np.array(["Station_A", "Station_A", "Station_B", "Station_B"])

    res_df = evaluate_by_subgroups(y_true, y_pred, groups)
    assert len(res_df) == 2
    assert "Station_A" in res_df.index
    assert "Station_B" in res_df.index
    assert res_df.loc["Station_A", "mae"] == 2.0
    assert res_df.loc["Station_B", "mae"] == pytest.approx(7.5, abs=1e-3)


class DummyModel:
    def predict(self, X):
        return np.ones(len(X)) * 50.0


def test_benchmark_latency():
    """Verify latency benchmarking utility."""
    model = DummyModel()
    X = np.random.randn(100, 10)
    bench = benchmark_latency(model, X, n_iterations=10)

    assert bench["batch_sample_count"] == 100
    assert bench["n_iterations"] == 10
    assert bench["mean_batch_latency_ms"] > 0.0
    assert bench["throughput_samples_per_sec"] > 0.0
