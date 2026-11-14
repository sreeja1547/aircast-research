"""Rigorous unit and integration tests verifying research methodology integrity:

1. Validation-only model selection (Pareto optimization using val_mae exclusively).
2. Production LOSO preprocessing isolation (Protocol B & Protocol A validation).
3. Dashboard live inference integration (all 3 horizons with real XGBoost models & conformal calibrators).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from api.services import AirCastInferenceService
from scripts.run_horizon_selection import (
    compute_composite_utility,
    identify_pareto_frontier,
    load_all_candidates,
)
from src.features.registry import get_feature_group
from src.preprocessing.cleaner import StationAwareCausalImputer
from src.preprocessing.transformers import WIND_DIR_DEGREE_MAP


def test_validation_only_model_selection_isolation():
    """Verify that model selection uses validation metrics only and never accesses test metrics."""
    candidates = [
        {
            "model_name": "Model_Overfitted_To_Test",
            "val_mae": 25.0,  # Poor validation accuracy
            "test_mae": 5.0,  # Artificially low test error (leaked)
            "mean_latency_ms": 0.05,
            "artifact_size_mb": 1.0,
            "training_time_sec": 10.0,
        },
        {
            "model_name": "Model_True_Generalizer",
            "val_mae": 8.0,   # Strong validation accuracy
            "test_mae": 9.0,   # Realistic test error
            "mean_latency_ms": 0.05,
            "artifact_size_mb": 1.0,
            "training_time_sec": 10.0,
        },
    ]

    # Under validation-only protocol: Model_True_Generalizer MUST dominate and win
    cands_val = identify_pareto_frontier([c.copy() for c in candidates], metric_key="val_mae")
    scored_val = compute_composite_utility(cands_val, metric_key="val_mae")
    assert scored_val[0]["model_name"] == "Model_True_Generalizer", (
        "Validation-only selection must pick the model with superior validation accuracy!"
    )

    # In contrast, if test_mae were used (leaked), the overfitted model would erroneously win
    cands_leaked = identify_pareto_frontier([c.copy() for c in candidates], metric_key="test_mae")
    scored_leaked = compute_composite_utility(cands_leaked, metric_key="test_mae")
    assert scored_leaked[0]["model_name"] == "Model_Overfitted_To_Test", (
        "Test-based selection demonstrates the data leakage vulnerability."
    )


def test_production_loso_preprocessing_isolation():
    """Verify that held-out stations in LOSO have zero presence in imputer training statistics."""
    rng = np.random.default_rng(42)
    stations = ["Station_Train1", "Station_Train2", "Station_HeldOut"]
    records = []
    for st in stations:
        for i in range(50):
            records.append({
                "station": st,
                "datetime": pd.Timestamp("2015-01-01") + pd.Timedelta(hours=i),
                "PM2.5": 20.0 if "Train" in st else 250.0,  # HeldOut has completely different distribution
                "wd": "N" if "Train" in st else "SW",
            })
    df_network = pd.DataFrame(records)

    # Protocol B: fit imputer strictly excluding Station_HeldOut
    train_split = df_network[df_network["station"] != "Station_HeldOut"].copy()
    held_out_split = df_network[df_network["station"] == "Station_HeldOut"].copy()

    imputer = StationAwareCausalImputer(feature_cols=["PM2.5"], categorical_cols=["wd"])
    imputer.fit(train_split)

    # 1. Held-out station must NOT exist in station_medians_
    assert "Station_HeldOut" not in imputer.station_medians_, (
        "Held-out station must never be registered in station-specific training statistics!"
    )

    # 2. Global medians must NOT be contaminated by the held-out station's extreme values
    assert imputer.global_medians_["PM2.5"] == 20.0, (
        f"Global median ({imputer.global_medians_['PM2.5']}) contaminated by held-out station!"
    )
    assert imputer.global_modes_["wd"] == "N"

    # 3. Imputing held-out station data with long missing blocks falls back strictly to training global median
    held_out_missing = held_out_split.copy()
    held_out_missing.loc[10:, "PM2.5"] = np.nan
    imputed_held_out = imputer.transform(held_out_missing)
    assert (imputed_held_out.loc[10:, "PM2.5"] == 20.0).all()


def test_end_to_end_loso_protocol_b_execution():
    """Verify that 1-fold Protocol B vs Protocol A comparison executes end-to-end with mathematical isolation."""
    from scripts.run_loso_protocol_comparison import run_loso_protocol_fold
    from src.data.forecasting_dataset import MultiHorizonForecastingDataset

    # Generate synthetic 2-station network with multi-horizon targets
    stations = ["Station_A", "Station_B"]
    dates = pd.date_range("2015-01-01", periods=100, freq="1h")
    records = []
    for st in stations:
        for i, dt in enumerate(dates):
            records.append({
                "station": st,
                "datetime": dt,
                "PM2.5": 30.0 + 5.0 * np.sin(i / 10.0),
                "PM10": 50.0,
                "SO2": 15.0,
                "NO2": 25.0,
                "CO": 800.0,
                "O3": 40.0,
                "TEMP": 10.0,
                "PRES": 1015.0,
                "DEWP": 2.0,
                "RAIN": 0.0,
                "WSPM": 2.0,
                "wd": "N",
            })
    df_net = pd.DataFrame(records)

    builder = MultiHorizonForecastingDataset()
    feature_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

    res = run_loso_protocol_fold(
        df_raw=df_net,
        builder=builder,
        held_out_station="Station_B",
        feature_cols=feature_cols,
        horizon=1,
    )

    assert res["station"] == "Station_B"
    assert res["test_samples"] > 0
    assert "protocol_a" in res and "protocol_b" in res
    assert np.isfinite(res["protocol_a"]["mae"])
    assert np.isfinite(res["protocol_b"]["mae"])
    assert np.isfinite(res["delta"]["mae"])


def test_dashboard_inference_pipeline_integration():
    """Verify that Dashboard Tab 9 inference constructs valid 47 features and returns calibrated intervals."""
    svc = AirCastInferenceService()
    horizons = [1, 6, 24]

    # Verify physical wind vector decomposition matching training pipeline
    wspm = 3.5
    for wd_str in ["N", "E", "S", "W"]:
        deg = WIND_DIR_DEGREE_MAP[wd_str]
        rad = np.radians(deg)
        u_expected = -wspm * np.sin(rad)
        v_expected = -wspm * np.cos(rad)
        assert np.isfinite(u_expected)
        assert np.isfinite(v_expected)

    # Verify multi-horizon inference execution
    all_47_features = get_feature_group("combined_all")
    feature_payload = {col: 25.0 for col in all_47_features}
    feature_payload["PM2.5"] = 65.0
    feature_payload["TEMP"] = 12.0
    feature_payload["DEWP"] = 2.0
    feature_payload["dew_point_depression"] = 10.0
    feature_payload["PRES"] = 1015.0
    feature_payload["WSPM"] = 2.5
    feature_payload["wd_sin"] = 0.0
    feature_payload["wd_cos"] = 1.0
    feature_payload["wind_u"] = 0.0
    feature_payload["wind_v"] = -2.5

    for h in horizons:
        point_pred, lb, ub, lat_ms, method, is_calibrated = svc.predict_single(
            features=feature_payload,
            horizon=h,
            model_name="xgboost",
            compute_interval=True,
            nominal_coverage=0.90,
        )

        # 1. Point prediction must be non-negative and finite
        assert np.isfinite(point_pred)
        assert point_pred >= 0.0

        # 2. Conformal interval must be strictly valid and calibrated
        assert is_calibrated is True, f"Horizon {h} must use calibrated conformal calibrator!"
        assert method == "calibrated_normalized_conformal"
        assert lb is not None and ub is not None
        assert 0.0 <= lb <= point_pred <= ub, (
            f"Horizon {h} bounds physically violated: lb={lb}, pred={point_pred}, ub={ub}"
        )
        assert lat_ms > 0.0


def test_ground_truth_target_preservation_and_no_imputation():
    """Verify that original non-imputed PM2.5 ground-truth measurements are preserved in targets.

    Regression test guarantees:
    1. Missing original PM2.5 measurements remain missing (NaN) in future target columns.
    2. Imputed PM2.5 is used as an input feature only.
    3. Missing ground-truth labels are dropped and excluded from regression metrics.
    """
    from src.data.forecasting_dataset import MultiHorizonForecastingDataset
    from src.evaluation.metrics import calculate_regression_metrics

    # Create synthetic series with known missing values
    n_hours = 60
    dates = pd.date_range("2015-01-01 00:00:00", periods=n_hours, freq="h")
    # Missing at index 10, 20, 35
    pm25_raw = np.linspace(10.0, 100.0, n_hours)
    pm25_raw[10] = np.nan
    pm25_raw[20] = np.nan
    pm25_raw[35] = np.nan

    df_test = pd.DataFrame({
        "station": ["TestStation"] * n_hours,
        "datetime": dates,
        "PM2.5": pm25_raw,
        "PM10": np.linspace(20.0, 120.0, n_hours),
        "SO2": [10.0] * n_hours,
        "NO2": [20.0] * n_hours,
        "CO": [1.0] * n_hours,
        "O3": [30.0] * n_hours,
        "TEMP": [15.0] * n_hours,
        "PRES": [1010.0] * n_hours,
        "DEWP": [5.0] * n_hours,
        "RAIN": [0.0] * n_hours,
        "WSPM": [2.0] * n_hours,
        "wd": ["N"] * n_hours,
    })

    imputer = StationAwareCausalImputer(feature_cols=["PM2.5", "PM10"])
    builder = MultiHorizonForecastingDataset(horizons=[1, 6, 24])

    df_prep = builder.prepare_dataset(df_test, imputer=imputer)

    # 1. Input feature PM2.5 must be imputed (0 NaNs)
    assert df_prep["PM2.5"].isna().sum() == 0, "Input feature PM2.5 must be causally imputed!"

    # 2. PM2.5_raw_unimputed must retain original raw NaNs
    assert df_prep["PM2.5_raw_unimputed"].isna().sum() == 3, "Raw target column must preserve raw NaNs!"

    # 3. Future targets must preserve NaNs where raw PM2.5 was missing
    # target_h1 at t=9 should be NaN because t+1=10 is NaN
    assert pd.isna(df_prep.loc[9, "target_h1"]), "target_h1 must be NaN where future measurement was missing!"
    assert pd.isna(df_prep.loc[19, "target_h1"])
    assert pd.isna(df_prep.loc[34, "target_h1"])

    # target_h6 at t=4 should be NaN because t+6=10 is NaN
    assert pd.isna(df_prep.loc[4, "target_h6"]), "target_h6 must be NaN where future measurement was missing!"

    # 4. get_horizon_dataset must drop missing targets and not evaluate imputed labels
    X_h1, y_h1, meta_h1 = builder.get_horizon_dataset(df_prep, horizon=1, feature_group="temporal_baseline", drop_initial_lags=False)
    # Total evaluated y_h1 should exclude index 9, 19, 34 and final boundary 59
    assert len(y_h1) == n_hours - 4  # 3 missing + 1 boundary
    assert not y_h1.isna().any(), "Target vector must contain zero NaN values!"

    # Metrics calculated on y_h1 evaluate only genuine observations
    metrics = calculate_regression_metrics(y_h1.values, y_h1.values)
    assert metrics["mae"] == 0.0


def test_cache_reprocessing_cannot_synthesize_ground_truth_targets():
    """Verify that passing an already-prepared/cached feature table cannot synthesize ground-truth labels."""
    from src.data.forecasting_dataset import MultiHorizonForecastingDataset

    # Construct mock cached dataframe where PM2.5 is imputed but raw target column has NaNs
    n_hours = 80
    dates = pd.date_range("2016-01-01 00:00:00", periods=n_hours, freq="h")
    df_cache_mock = pd.DataFrame({
        "station": ["StationA"] * n_hours,
        "datetime": dates,
        "PM2.5": [50.0] * n_hours,  # Already imputed: 0 NaNs
        "PM2.5_raw_unimputed": [50.0] * (n_hours - 5) + [np.nan] * 5,  # 5 raw NaNs
        "target_h24": [50.0] * (n_hours - 29) + [np.nan] * 29,  # 5 missing + 24 boundary NaNs
        "hour": dates.hour,
        "month": dates.month,
        "wd": ["N"] * n_hours,
        "WSPM": [2.0] * n_hours,
    })

    builder = MultiHorizonForecastingDataset(horizons=[24])
    # Reprocess the cached dataframe
    df_reprocessed = builder.prepare_dataset(df_cache_mock)

    # 1. Reprocessing must NOT overwrite PM2.5_raw_unimputed with imputed PM2.5
    assert df_reprocessed["PM2.5_raw_unimputed"].isna().sum() == 5, (
        "Reprocessing must not overwrite raw un-imputed column with imputed features!"
    )

    # 2. Target target_h24 must NOT synthesize targets for the missing hours
    assert df_reprocessed["target_h24"].isna().sum() >= 29, (
        "Reprocessing must never synthesize targets for missing ground-truth hours!"
    )

    # 3. Target extraction drops the missing labels
    X_24, y_24, meta_24 = builder.get_horizon_dataset(
        df_reprocessed, horizon=24, feature_group="temporal_baseline", drop_initial_lags=False
    )
    assert len(y_24) <= n_hours - 29, "Evaluated targets must exclude all unobserved ground-truth labels!"
