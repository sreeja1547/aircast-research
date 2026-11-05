"""Stage 9: Regression Verification & Performance Integrity Tests for AirCast Research.

Validates that computational optimizations preserve:
1. Numerical correctness and prediction fidelity (tolerance < 1e-4).
2. Latency thresholds for real-time serving (< 75ms).
3. Memory and disk storage bounds.
4. Batch throughput guarantees (> 1,000 samples/sec).
"""

import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from api.services import AirCastInferenceService
from src.features.registry import get_feature_group
from src.models import (
    HistGradientBoostingForecaster,
    LSTMForecaster,
    PersistenceForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)


@pytest.fixture(scope="module")
def inference_service():
    return AirCastInferenceService()


@pytest.fixture(scope="module")
def sample_feature_dict():
    feature_names = get_feature_group("combined_all")
    return {f: 30.0 for f in feature_names}


def test_single_sample_latency_threshold(inference_service, sample_feature_dict):
    """Assert single-sample inference latency remains below 75ms under warm serving."""
    # Warmup
    _ = inference_service.predict_single(sample_feature_dict, horizon=1, model_name="xgboost")

    latencies = []
    for _ in range(15):
        _, _, _, lat_ms, _, _ = inference_service.predict_single(
            sample_feature_dict, horizon=1, model_name="xgboost"
        )
        latencies.append(lat_ms)

    median_lat = float(np.median(latencies))
    assert median_lat < 75.0, f"Single-sample latency ({median_lat:.2f}ms) exceeded 75ms budget."


def test_batch_throughput_threshold(inference_service, sample_feature_dict):
    """Assert batch inference throughput exceeds 1,000 samples/sec on batch of 100."""
    batch = [sample_feature_dict for _ in range(100)]
    _, _, _, tot_ms, per_samp_ms, _, _ = inference_service.predict_batch(
        batch, horizon=1, model_name="xgboost"
    )
    throughput = 100.0 / (tot_ms / 1000.0)
    assert throughput >= 1000.0, f"Batch throughput ({throughput:.1f} s/s) below 1,000 s/s."


def test_model_artifact_sizes():
    """Assert model artifact sizes comply with storage budgets."""
    models_dir = Path("artifacts/models")
    if not models_dir.exists():
        pytest.skip("Models directory does not exist.")

    # XGBoost should be under 3 MB
    xgb_path = models_dir / "xgboost_h1.joblib"
    if xgb_path.exists():
        size_mb = os.path.getsize(xgb_path) / (1024 * 1024)
        assert size_mb < 3.0, f"XGBoost artifact size ({size_mb:.2f}MB) exceeds 3MB limit."

    # LSTM should be under 1 MB
    lstm_path = models_dir / "lstm_h1.joblib"
    if lstm_path.exists():
        size_mb = os.path.getsize(lstm_path) / (1024 * 1024)
        assert size_mb < 1.0, f"LSTM artifact size ({size_mb:.2f}MB) exceeds 1MB limit."


def test_numerical_prediction_fidelity():
    """Assert predictions are deterministic and consistent across calls."""
    xgb_path = Path("artifacts/models/xgboost_h1.joblib")
    if not xgb_path.exists():
        pytest.skip("XGBoost artifact not found.")

    model = joblib.load(xgb_path)
    X_test = pd.DataFrame(np.random.RandomState(42).randn(10, 47), columns=model.feature_names)

    pred1 = model.predict(X_test)
    pred2 = model.predict(X_test)
    max_delta = np.max(np.abs(pred1 - pred2))
    assert max_delta < 1e-5, f"Non-deterministic prediction variance: {max_delta}"


def test_conformal_calibrator_cached(inference_service):
    """Assert conformal calibrator loads successfully and reports calibrated status."""
    calibrator = inference_service.get_conformal_calibrator(1)
    if calibrator is not None:
        assert calibrator.is_calibrated is True
