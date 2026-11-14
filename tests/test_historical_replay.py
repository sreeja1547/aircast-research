"""Automated tests for dashboard historical replay, columnar loading, and station encoding."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from api.services import AirCastInferenceService
from dashboard.app import load_historical_test_record_cached
from src.features.registry import get_feature_group
from src.models.base import BaseForecaster

CACHE_PATH = "data/processed/forecasting_features_cache.parquet"

PRESET_EPISODES = {
    "Severe Winter Smog (Tiantan, 2016-10-02 21:00)": ("2016-10-02 21:00:00", "Tiantan", 9.0),
    "Extreme Inversion Event (Dongsi, 2017-01-01 18:00)": ("2017-01-01 18:00:00", "Dongsi", 3.0),
    "Moderate Urban Episode (Aotizhongxin, 2016-11-20 09:00)": ("2016-11-20 09:00:00", "Aotizhongxin", 0.0),
    "Clean Post-Frontal Day (Huairou, 2016-10-15 12:00)": ("2016-10-15 12:00:00", "Huairou", 6.0),
}


def test_station_encoding_matches_training_base_model():
    """Assert station categorical index mapping matches BaseForecaster._validate_inputs()."""
    class ConcreteForecaster(BaseForecaster):
        def fit(self, X, y=None):
            self._validate_inputs(X, y, is_training=True)
            self.is_fitted = True
            return self

        def predict(self, X):
            return np.zeros(len(X))

    stations_12 = [
        "Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng",
        "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"
    ]
    # Create mock training df to observe BaseForecaster internal mapping
    df_mock = pd.DataFrame({"station": stations_12, "PM2.5": [10.0] * 12})
    forecaster = ConcreteForecaster(name="test_encoding", model_type="baseline")
    X_enc = forecaster._validate_inputs(df_mock, is_training=True)

    encoder_map = forecaster.categorical_encoders_["station"]
    for i, st in enumerate(stations_12):
        assert encoder_map[st] == float(i), f"Encoding mismatch for station {st}"
        assert X_enc["station"].iloc[i] == float(i)


def test_historical_replay_extraction_schema_and_types():
    """Verify historical replay extraction yields valid 47-feature vectors without NaNs or strings."""
    if not os.path.exists(CACHE_PATH):
        pytest.skip("Parquet cache not found on disk")

    expected_47 = set(get_feature_group("combined_all"))
    assert len(expected_47) == 47

    for ep_name, (dt_str, st_name, expected_st_idx) in PRESET_EPISODES.items():
        rec = load_historical_test_record_cached(CACHE_PATH, dt_str, st_name)
        assert rec is not None, f"Failed loading preset episode: {ep_name}"
        assert len(rec) == 47, f"Expected 47 features for {ep_name}, got {len(rec)}"
        assert set(rec.keys()) == expected_47, f"Feature key mismatch in {ep_name}"

        # Station encoding check
        assert rec["station"] == expected_st_idx, f"Station index {rec['station']} != {expected_st_idx}"

        # Value cleanliness: finite floats, non-null
        for col, val in rec.items():
            assert isinstance(val, float), f"Column {col} in {ep_name} is not float: {type(val)}"
            assert np.isfinite(val), f"Column {col} in {ep_name} has non-finite value: {val}"


def test_historical_replay_inference_compatibility():
    """Verify extracted genuine test records pass full AirCast inference service validation."""
    if not os.path.exists(CACHE_PATH):
        pytest.skip("Parquet cache not found on disk")

    svc = AirCastInferenceService()
    dt_str, st_name, _ = PRESET_EPISODES["Severe Winter Smog (Tiantan, 2016-10-02 21:00)"]
    rec = load_historical_test_record_cached(CACHE_PATH, dt_str, st_name)
    assert rec is not None

    for h in [1, 6, 24]:
        pred, lb, ub, lat, method, is_cal = svc.predict_single(rec, horizon=h, model_name="recommended")
        assert pred > 0.0
        assert lb is not None and ub is not None
        assert 0.0 <= lb <= pred <= ub
        assert is_cal is True
        assert method == "calibrated_normalized_conformal"


def test_missing_record_returns_none_cleanly():
    """Verify non-existent timestamps or stations return None without throwing exceptions."""
    if not os.path.exists(CACHE_PATH):
        pytest.skip("Parquet cache not found on disk")

    res = load_historical_test_record_cached(CACHE_PATH, "1999-01-01 00:00:00", "Tiantan")
    assert res is None

    res_st = load_historical_test_record_cached(CACHE_PATH, "2016-10-02 21:00:00", "NonExistentStation")
    assert res_st is None
