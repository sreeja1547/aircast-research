"""Automated test suite for preprocessing pipelines.

Verifies:
1. Station-aware causal imputation (no lookahead / zero future leakage).
2. Strict target preservation (missing PM2.5 targets are dropped, never imputed).
3. 24-hour boundary purge between temporal partitions.
4. Station holdout spatial isolation.
5. Cyclical and wind vector trigonometric correctness.
6. Scaler fit-transform isolation (fitted only on training partition).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.preprocessing.cleaner import StationAwareCausalImputer, drop_unobserved_targets
from src.preprocessing.splitter import TemporalPurgedSplitter, StationHoldoutSplitter
from src.preprocessing.transformers import (
    CyclicalTransformer,
    WindVectorTransformer,
    RobustFeatureScaler,
    WIND_DIR_DEGREE_MAP,
)


@pytest.fixture
def synthetic_station_series():
    """Create a controlled mini-timeseries with known missingness patterns."""
    dates = pd.date_range("2015-01-01 00:00:00", periods=48, freq="1h")
    df = pd.DataFrame({
        "No": range(1, 49),
        "datetime": dates,
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "hour": dates.hour,
        "station": ["Aotizhongxin"] * 48,
        "PM2.5": [10.0 + i for i in range(48)],
        "TEMP": [0.0 + (i % 10) for i in range(48)],
        "wd": ["N", "NE", "E", "SE"] * 12,
        "WSPM": [2.0] * 48,
    })
    # Inject missing values
    df.loc[5:8, "PM2.5"] = np.nan  # Gap of 4 hours
    df.loc[0, "TEMP"] = np.nan     # Missing at the very start
    return df


def test_causal_imputer_no_lookahead(synthetic_station_series):
    """Test that imputation only uses past values and never peeks ahead."""
    df = synthetic_station_series.copy()
    imputer = StationAwareCausalImputer(feature_cols=["PM2.5", "TEMP"], add_indicators=True)
    imputer.fit(df)
    transformed = imputer.transform(df)

    # Value at index 4 was 14.0 (10 + 4).
    # Since index 5-8 were NaN, causal forward fill must propagate 14.0 into rows 5, 6, 7, 8.
    # It must NOT look ahead to row 9 (which is 19.0).
    for idx in range(5, 9):
        assert transformed.loc[idx, "PM2.5"] == 14.0, f"Expected 14.0 at index {idx}, got {transformed.loc[idx, 'PM2.5']}"

    # Missing indicators must be 1.0 for imputed rows and 0.0 otherwise
    assert (transformed.loc[5:8, "PM2.5_missing"] == 1.0).all()
    assert (transformed.loc[9:15, "PM2.5_missing"] == 0.0).all()


def test_drop_unobserved_targets():
    """Test that missing PM2.5 target rows are strictly dropped and never imputed."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2015-01-01", periods=10, freq="1h"),
        "target_h1": [10.0, np.nan, 30.0, np.nan, 50.0, 60.0, np.nan, 80.0, 90.0, 100.0]
    })
    cleaned = drop_unobserved_targets(df, "target_h1")

    assert len(cleaned) == 7
    assert cleaned["target_h1"].isnull().sum() == 0
    assert list(cleaned["target_h1"]) == [10.0, 30.0, 50.0, 60.0, 80.0, 90.0, 100.0]


def test_temporal_purged_splitter():
    """Verify that a 24-hour purge buffer prevents target overlap between splits."""
    dates = pd.date_range("2015-12-30 00:00:00", "2016-10-10 23:00:00", freq="1h")
    df = pd.DataFrame({
        "datetime": dates,
        "station": ["Aotizhongxin"] * len(dates),
        "val": range(len(dates))
    })

    splitter = TemporalPurgedSplitter(
        train_end="2015-12-31 23:00:00",
        val_start="2016-01-02 00:00:00",
        val_end="2016-09-30 23:00:00",
        test_start="2016-10-02 00:00:00",
        test_end="2016-10-10 23:00:00",
    )

    train_df, val_df, test_df = splitter.split(df)

    # 1. Check max timestamp in train vs min timestamp in val
    train_max = train_df["datetime"].max()
    val_min = val_df["datetime"].min()
    gap_train_val = (val_min - train_max).total_seconds() / 3600.0
    assert gap_train_val >= 25.0, f"Gap must be >= 25 hours between train max and val min (24h full day purge), got {gap_train_val}"

    # 2. Check max timestamp in val vs min timestamp in test
    val_max = val_df["datetime"].max()
    test_min = test_df["datetime"].min()
    gap_val_test = (test_min - val_max).total_seconds() / 3600.0
    assert gap_val_test >= 25.0, f"Gap must be >= 25 hours between val max and test min, got {gap_val_test}"

    # 3. Check mutual exclusivity
    train_dates = set(train_df["datetime"])
    val_dates = set(val_df["datetime"])
    test_dates = set(test_df["datetime"])
    assert len(train_dates.intersection(val_dates)) == 0
    assert len(val_dates.intersection(test_dates)) == 0


def test_station_holdout_splitter():
    """Verify that held out station is completely excluded from training partition."""
    df = pd.DataFrame({
        "station": ["Aotizhongxin", "Changping", "Dingling", "Dongsi"] * 10,
        "value": range(40)
    })
    splitter = StationHoldoutSplitter()
    train_df, test_df = splitter.split_by_station(df, held_out_station="Dingling")

    assert "Dingling" not in train_df["station"].unique()
    assert (test_df["station"] == "Dingling").all()
    assert len(train_df) + len(test_df) == len(df)


def test_cyclical_transformer():
    """Verify trigonometric encoding of hour and month."""
    df = pd.DataFrame({
        "hour": [0, 6, 12, 18, 23],
        "month": [1, 4, 7, 10, 12]
    })
    transformer = CyclicalTransformer()
    res = transformer.transform(df)

    # At hour 0: sin(0) = 0, cos(0) = 1
    assert np.isclose(res.loc[0, "hour_sin"], 0.0, atol=1e-5)
    assert np.isclose(res.loc[0, "hour_cos"], 1.0, atol=1e-5)

    # At hour 6: sin(pi/2) = 1, cos(pi/2) = 0
    assert np.isclose(res.loc[1, "hour_sin"], 1.0, atol=1e-5)
    assert np.isclose(res.loc[1, "hour_cos"], 0.0, atol=1e-5)

    # Bounds check
    assert (res["hour_sin"] >= -1.0).all() and (res["hour_sin"] <= 1.0).all()
    assert (res["hour_cos"] >= -1.0).all() and (res["hour_cos"] <= 1.0).all()


def test_wind_vector_transformer():
    """Verify physical Cartesian wind decomposition."""
    df = pd.DataFrame({
        "wd": ["N", "E", "S", "W"],
        "WSPM": [10.0, 10.0, 10.0, 10.0]
    })
    transformer = WindVectorTransformer()
    res = transformer.transform(df)

    # North wind (0 deg): blows from North towards South -> u = 0, v = -10
    assert np.isclose(res.loc[0, "wind_u"], 0.0, atol=1e-5)
    assert np.isclose(res.loc[0, "wind_v"], -10.0, atol=1e-5)

    # East wind (90 deg): blows from East towards West -> u = -10, v = 0
    assert np.isclose(res.loc[1, "wind_u"], -10.0, atol=1e-5)
    assert np.isclose(res.loc[1, "wind_v"], 0.0, atol=1e-5)


def test_scaler_fitted_on_train_only():
    """Verify that scaler parameters are derived solely from training partition."""
    train_df = pd.DataFrame({"PM2.5": [10.0, 20.0, 30.0, 40.0, 50.0]})
    test_df = pd.DataFrame({"PM2.5": [100.0, 200.0, 300.0]})

    scaler = RobustFeatureScaler(feature_cols=["PM2.5"])
    scaler.fit(train_df)

    # Train median = 30.0, IQR = 40.0 - 20.0 = 20.0
    assert scaler.medians_["PM2.5"] == 30.0
    assert scaler.iqrs_["PM2.5"] == 20.0

    # Transform test set using TRAIN parameters:
    # 100 -> (100 - 30)/20 = 3.5
    test_scaled = scaler.transform(test_df)
    assert np.isclose(test_scaled.loc[0, "PM2.5"], 3.5)
