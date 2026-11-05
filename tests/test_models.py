"""Unit tests for AirCast forecasting models and model registry.

Verifies the fit/predict contract, deterministic behavior, non-negativity constraint,
feature importances, native TreeSHAP computation, and persistence round-trip.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models import (
    MODEL_REGISTRY,
    BaseForecaster,
    HistGradientBoostingForecaster,
    LinearRegressionForecaster,
    LSTMForecaster,
    PersistenceForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
    get_model,
    list_available_models,
)


@pytest.fixture
def synthetic_data():
    """Create reproducible synthetic forecasting data."""
    np.random.seed(42)
    n = 200
    features = {
        "PM2.5": np.random.uniform(10.0, 200.0, n),
        "PM10": np.random.uniform(20.0, 300.0, n),
        "SO2": np.random.uniform(2.0, 50.0, n),
        "NO2": np.random.uniform(10.0, 80.0, n),
        "TEMP": np.random.uniform(-10.0, 35.0, n),
        "WSPM": np.random.uniform(0.5, 8.0, n),
    }
    df_X = pd.DataFrame(features)
    # Target PM2.5 at t+h with physical relationship and noise
    y = (
        0.7 * df_X["PM2.5"]
        + 0.1 * df_X["PM10"]
        - 2.5 * df_X["WSPM"]
        + np.random.normal(0, 5, n)
    )
    y = np.maximum(0.0, y)

    X_train, X_val = df_X.iloc[:150], df_X.iloc[150:]
    y_train, y_val = y.iloc[:150], y.iloc[150:]

    return X_train, y_train, X_val, y_val


def test_registry_contains_all_models():
    """Verify registry catalog completeness and factory instantiation."""
    expected_models = [
        "persistence",
        "linear_regression",
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
        "lstm",
    ]
    available = list_available_models()
    for m in expected_models:
        assert m in available
        instance = get_model(m)
        assert isinstance(instance, BaseForecaster)


def test_persistence_forecaster(synthetic_data):
    """Verify Persistence baseline extracts exact current PM2.5 values."""
    X_train, y_train, X_val, y_val = synthetic_data

    model = PersistenceForecaster(target_col="PM2.5")
    model.fit(X_train, y_train)
    assert model.is_fitted
    assert model.resolved_col_ == "PM2.5"

    preds = model.predict(X_val)
    assert len(preds) == len(X_val)
    # Must equal exact X_val['PM2.5']
    np.testing.assert_allclose(preds, X_val["PM2.5"].to_numpy())
    assert np.all(preds >= 0.0)


def test_linear_and_ridge_forecasters(synthetic_data, tmp_path):
    """Verify Linear Regression and Ridge forecasters with scaling and serialization."""
    X_train, y_train, X_val, _ = synthetic_data

    for ModelClass, name in [(LinearRegressionForecaster, "lr"), (RidgeForecaster, "ridge")]:
        model = ModelClass(name=name, scaler_type="robust")
        model.fit(X_train, y_train)
        assert model.is_fitted

        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        assert np.all(preds >= 0.0)  # Physical non-negativity

        # Test persistence
        save_path = tmp_path / f"{name}_model.joblib"
        model.save(save_path)
        loaded = ModelClass.load(save_path)
        loaded_preds = loaded.predict(X_val)
        np.testing.assert_allclose(preds, loaded_preds)


def test_random_forest_forecaster(synthetic_data, tmp_path):
    """Verify Random Forest forecaster and feature importances."""
    X_train, y_train, X_val, _ = synthetic_data

    model = RandomForestForecaster(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    assert model.is_fitted

    preds = model.predict(X_val)
    assert len(preds) == len(X_val)
    assert np.all(preds >= 0.0)

    importances = model.get_feature_importances()
    assert len(importances) == X_train.shape[1]
    assert importances.index[0] in X_train.columns  # Top feature is from schema
    assert np.isclose(importances.sum(), 1.0)

    # Persistence round-trip
    save_path = tmp_path / "rf_model.joblib"
    model.save(save_path)
    loaded = RandomForestForecaster.load(save_path)
    loaded_preds = loaded.predict(X_val)
    np.testing.assert_allclose(preds, loaded_preds)


def test_hist_gradient_boosting_forecaster(synthetic_data):
    """Verify native HistGradientBoosting forecaster."""
    X_train, y_train, X_val, _ = synthetic_data

    model = HistGradientBoostingForecaster(max_iter=20, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    assert model.is_fitted

    preds = model.predict(X_val)
    assert len(preds) == len(X_val)
    assert np.all(preds >= 0.0)


def test_xgboost_forecaster_and_treeshap(synthetic_data, tmp_path):
    """Verify XGBoost with early stopping, feature importances, and native TreeSHAP."""
    X_train, y_train, X_val, y_val = synthetic_data

    model = XGBoostForecaster(
        n_estimators=50,
        max_depth=4,
        early_stopping_rounds=5,
        random_state=42,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert model.is_fitted

    preds = model.predict(X_val)
    assert len(preds) == len(X_val)
    assert np.all(preds >= 0.0)

    # Feature importances
    gain_imp = model.get_feature_importances(importance_type="gain")
    assert len(gain_imp) == X_train.shape[1]

    # Native TreeSHAP computation
    shap_out = model.compute_shap_values(X_val.iloc[:10])
    shap_vals = shap_out["shap_values"]
    base_val = shap_out["base_value"]
    assert shap_vals.shape == (10, X_train.shape[1])
    assert isinstance(base_val, float)

    # Additive efficiency check: sum of SHAP values + base_val = raw margin
    # (for squarederror objective, raw margin is the prediction before clipping)
    raw_pred_sample = model.model.predict(X_val.iloc[:10])
    reconstructed = np.sum(shap_vals, axis=1) + base_val
    np.testing.assert_allclose(raw_pred_sample, reconstructed, rtol=1e-4)

    # Persistence round-trip
    save_path = tmp_path / "xgb_model.joblib"
    model.save(save_path)
    loaded = XGBoostForecaster.load(save_path)
    loaded_preds = loaded.predict(X_val)
    np.testing.assert_allclose(preds, loaded_preds)


def test_lstm_forecaster(synthetic_data, tmp_path):
    """Verify PyTorch LSTM forecaster with early stopping and state serialization."""
    X_train, y_train, X_val, y_val = synthetic_data

    model = LSTMForecaster(
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        max_epochs=5,
        patience=2,
        batch_size=32,
        random_state=42,
        device="cpu",
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert model.is_fitted

    preds = model.predict(X_val)
    assert len(preds) == len(X_val)
    assert np.all(preds >= 0.0)
    assert not np.isnan(preds).any()

    # Persistence round-trip
    save_path = tmp_path / "lstm_model.joblib"
    model.save(save_path)
    loaded = LSTMForecaster.load(save_path)
    loaded_preds = loaded.predict(X_val)
    np.testing.assert_allclose(preds, loaded_preds, rtol=1e-4)
