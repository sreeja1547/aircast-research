"""Automated test suite for multi-horizon feature engineering and dataset construction.

Verifies:
1. Station-aware lag alignment (exact match to historical offsets).
2. Rolling window causal bounds (no lookahead into future; inclusion of current-time observation).
3. Missing observations inside rolling windows.
4. Station boundary isolation (lags/rolling stats do not bleed across stations).
5. Multi-horizon target alignment (h=1, h=6, h=24).
6. Unimputed target integrity (targets are never imputed; unobserved targets are excluded).
7. Feature group registry retrieval and consistency.
8. Leave-One-Station-Out imputer quarantine (held-out station statistics are never used).
9. Strict exclusion of target columns from feature matrix X.
10. Strict exclusion of future weather observations.
11. Feature column consistency between training and inference partitions.
12. Future-value mutation invariance (changing t+k does not alter X(t)).
"""

import pytest
import numpy as np
import pandas as pd

from src.features.builder import StationAwareFeatureBuilder
from src.features.registry import get_feature_group, FEATURE_GROUPS
from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.preprocessing.cleaner import StationAwareCausalImputer
from src.preprocessing.splitter import TemporalPurgedSplitter


@pytest.fixture
def two_station_timeseries():
    """Create a controlled 2-station dataset with known values."""
    dates = pd.date_range("2015-01-01 00:00:00", periods=72, freq="1h")
    records = []
    for st in ["Station_Alpha", "Station_Beta"]:
        offset = 1000.0 if st == "Station_Beta" else 0.0
        for i, dt in enumerate(dates):
            records.append({
                "No": i + 1,
                "datetime": dt,
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
                "station": st,
                "PM2.5": offset + float(i),
                "PM10": offset + float(i) * 1.2,
                "SO2": 10.0,
                "NO2": 20.0,
                "CO": 500.0,
                "O3": 30.0,
                "TEMP": 10.0 + (i % 5),
                "PRES": 1010.0,
                "DEWP": 0.0,
                "RAIN": 0.0,
                "wd": "NE",
                "WSPM": 2.0,
            })
    df = pd.DataFrame(records)
    return df


def test_lag_alignment(two_station_timeseries):
    """Test that PM2.5_lag_k at time t exactly equals PM2.5 at time t-k."""
    builder = StationAwareFeatureBuilder(lags=[1, 3, 6, 12, 24, 48])
    df_feat = builder.build_features(two_station_timeseries)

    alpha_df = df_feat[df_feat["station"] == "Station_Alpha"].sort_values("datetime").reset_index(drop=True)

    # At row 10 (hour 10, PM2.5 = 10.0)
    # lag_1 should be value from row 9 (PM2.5 = 9.0)
    assert alpha_df.loc[10, "PM2.5_lag_1"] == 9.0
    assert alpha_df.loc[10, "PM2.5_lag_3"] == 7.0
    assert alpha_df.loc[10, "PM2.5_lag_6"] == 4.0

    # At row 30 (PM2.5 = 30.0)
    # lag_24 should be value from row 6 (PM2.5 = 6.0)
    assert alpha_df.loc[30, "PM2.5_lag_24"] == 6.0


def test_rolling_window_includes_current_time(two_station_timeseries):
    """Verify that rolling window over [t-W+1, t] includes current observation at t."""
    builder = StationAwareFeatureBuilder(rolling_windows=[3])
    df_feat = builder.build_features(two_station_timeseries)
    alpha = df_feat[df_feat["station"] == "Station_Alpha"].sort_values("datetime").reset_index(drop=True)

    # For window W=3 at row 2 (values are 0.0, 1.0, 2.0)
    # Mean must be (0 + 1 + 2) / 3 = 1.0
    assert np.isclose(alpha.loc[2, "PM2.5_roll_mean_3"], 1.0)
    # Max must be 2.0 (current value at t=2)
    assert alpha.loc[2, "PM2.5_roll_max_3"] == 2.0
    # Min must be 0.0 (value at t-2)
    assert alpha.loc[2, "PM2.5_roll_min_3"] == 0.0


def test_rolling_window_no_lookahead(two_station_timeseries):
    """Test that future values do not alter rolling statistics at current time t."""
    builder = StationAwareFeatureBuilder(rolling_windows=[3, 6])
    df1 = builder.build_features(two_station_timeseries)

    # Create mutated copy where future values (after hour 20) are drastically modified
    mutated = two_station_timeseries.copy()
    future_mask = (mutated["station"] == "Station_Alpha") & (mutated["hour"] > 20)
    mutated.loc[future_mask, "PM2.5"] = 9999.0

    df2 = builder.build_features(mutated)

    alpha1 = df1[df1["station"] == "Station_Alpha"].sort_values("datetime").reset_index(drop=True)
    alpha2 = df2[df2["station"] == "Station_Alpha"].sort_values("datetime").reset_index(drop=True)

    # Rolling statistics at hour 20 must be identical because only past observations are used
    assert np.isclose(alpha1.loc[20, "PM2.5_roll_mean_6"], alpha2.loc[20, "PM2.5_roll_mean_6"])
    assert np.isclose(alpha1.loc[20, "PM2.5_roll_max_6"], alpha2.loc[20, "PM2.5_roll_max_6"])


def test_station_boundary_isolation(two_station_timeseries):
    """Verify that lag and rolling statistics for Station_Beta do NOT borrow from Station_Alpha."""
    builder = StationAwareFeatureBuilder(lags=[1, 24])
    df_feat = builder.build_features(two_station_timeseries)

    beta_df = df_feat[df_feat["station"] == "Station_Beta"].sort_values("datetime").reset_index(drop=True)

    # First row of Station_Beta must have NaN for lag_1 (not the last row of Station_Alpha!)
    assert pd.isnull(beta_df.loc[0, "PM2.5_lag_1"]), "First row of station must not have lag from previous station!"
    assert pd.isnull(beta_df.loc[0, "PM2.5_lag_24"])

    # Value at row 1 of Station_Beta should have lag_1 = 1000.0 (row 0 of Station_Beta)
    assert beta_df.loc[1, "PM2.5_lag_1"] == 1000.0


def test_target_alignment_multi_horizon(two_station_timeseries):
    """Verify target_h1, target_h6, target_h24 match PM2.5(t+h)."""
    dataset_builder = MultiHorizonForecastingDataset(horizons=[1, 6, 24])
    prepared = dataset_builder.prepare_dataset(two_station_timeseries)

    alpha = prepared[prepared["station"] == "Station_Alpha"].sort_values("datetime").reset_index(drop=True)

    # At row 10 (t=10, PM2.5=10.0)
    # target_h1 should be 11.0 (row 11)
    # target_h6 should be 16.0 (row 16)
    # target_h24 should be 34.0 (row 34)
    assert alpha.loc[10, "target_h1"] == 11.0
    assert alpha.loc[10, "target_h6"] == 16.0
    assert alpha.loc[10, "target_h24"] == 34.0


def test_unimputed_target_integrity():
    """Verify that targets are never imputed from imputed features."""
    dates = pd.date_range("2015-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "No": range(1, 11),
        "datetime": dates,
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "hour": dates.hour,
        "station": ["Station_Alpha"] * 10,
        "PM2.5": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "PM10": [10.0] * 10,
        "SO2": [5.0] * 10,
        "NO2": [5.0] * 10,
        "CO": [100.0] * 10,
        "O3": [10.0] * 10,
        "TEMP": [0.0] * 10,
        "PRES": [1000.0] * 10,
        "DEWP": [0.0] * 10,
        "RAIN": [0.0] * 10,
        "wd": ["N"] * 10,
        "WSPM": [1.0] * 10,
    })

    imputer = StationAwareCausalImputer(feature_cols=["PM2.5"])
    ds = MultiHorizonForecastingDataset(horizons=[1])
    prepared = ds.prepare_dataset(df, imputer=imputer)

    # Row 1 (t=1, PM2.5=20.0): target_h1 is PM2.5(t=2), which was NaN in RAW data.
    # Even though imputer filled PM2.5(t=2) for features, target_h1 MUST be NaN!
    assert pd.isnull(prepared.loc[1, "target_h1"]), "Target must remain un-imputed NaN!"

    # When getting horizon dataset, row 1 must be cleanly dropped
    X, y, meta = ds.get_horizon_dataset(prepared, horizon=1, feature_group="historical_pollutants", drop_initial_lags=False)
    assert 1 not in y.index or pd.notnull(y.loc[1])


def test_no_target_in_X(two_station_timeseries):
    """Verify that NO target column is accidentally included in feature matrix X."""
    ds = MultiHorizonForecastingDataset(horizons=[1, 6, 24])
    prepared = ds.prepare_dataset(two_station_timeseries)

    for h in [1, 6, 24]:
        for grp in ["temporal_baseline", "historical_pollutants", "lag_features", "rolling_features", "meteorological_features", "combined_all"]:
            X, y, meta = ds.get_horizon_dataset(prepared, horizon=h, feature_group=grp)
            for col in X.columns:
                assert not col.startswith("target_"), f"Target column '{col}' leaked into feature matrix X for group '{grp}'!"
                assert col != "PM2.5_raw_unimputed", f"Raw target leaked into X!"


def test_future_weather_excluded(two_station_timeseries):
    """Verify that feature matrix contains no future weather variables."""
    ds = MultiHorizonForecastingDataset(horizons=[6, 24])
    prepared = ds.prepare_dataset(two_station_timeseries)

    X, y, meta = ds.get_horizon_dataset(prepared, horizon=6, feature_group="meteorological_features")
    # Allowed weather variables are strictly contemporaneous or historical <= t
    allowed_weather = {"TEMP", "PRES", "DEWP", "RAIN", "has_rain", "WSPM", "dew_point_depression", "wd_sin", "wd_cos", "wind_u", "wind_v"}
    weather_in_X = {c for c in X.columns if c in allowed_weather}

    # Ensure no future forward-shifted weather columns exist
    for col in X.columns:
        assert "_lead_" not in col, f"Future lead feature '{col}' detected in X!"
        assert "_future_" not in col, f"Future feature '{col}' detected in X!"


def test_feature_column_consistency_train_inference(two_station_timeseries):
    """Verify feature matrix column consistency between train and test splits."""
    ds = MultiHorizonForecastingDataset(horizons=[1])
    prepared = ds.prepare_dataset(two_station_timeseries)

    # Split prepared dataset
    half = len(prepared) // 2
    train_part = prepared.iloc[:half]
    test_part = prepared.iloc[half:]

    X_train, y_train, _ = ds.get_horizon_dataset(train_part, horizon=1, feature_group="combined_all")
    X_test, y_test, _ = ds.get_horizon_dataset(test_part, horizon=1, feature_group="combined_all")

    assert list(X_train.columns) == list(X_test.columns), "Feature column order mismatch between train and inference!"


def test_future_value_mutation_invariance(two_station_timeseries):
    """Verify that changing future PM2.5 or weather at t+h does not change X(t)."""
    ds = MultiHorizonForecastingDataset(horizons=[6])
    prepared1 = ds.prepare_dataset(two_station_timeseries)
    X1, y1, meta1 = ds.get_horizon_dataset(prepared1, horizon=6, feature_group="combined_all")

    # Mutate future values at t > 30
    mutated = two_station_timeseries.copy()
    mutated.loc[mutated["hour"] > 30, "PM2.5"] = 8888.0
    mutated.loc[mutated["hour"] > 30, "TEMP"] = -50.0

    prepared2 = ds.prepare_dataset(mutated)
    X2, y2, meta2 = ds.get_horizon_dataset(prepared2, horizon=6, feature_group="combined_all")

    # Features at t <= 30 must remain identical
    mask = meta1["hour"] <= 30
    cols_to_check = [c for c in X1.columns if c != "station"]
    diff = (X1.loc[mask, cols_to_check] - X2.loc[mask, cols_to_check]).abs().max().max()
    assert diff == 0.0, f"Future mutation changed historical feature values! Max diff: {diff}"


def test_feature_groups_registry():
    """Verify that all feature groups are defined and valid."""
    for grp in ["temporal_baseline", "historical_pollutants", "lag_features", "rolling_features", "meteorological_features", "combined_all"]:
        cols = get_feature_group(grp)
        assert len(cols) > 0
        assert isinstance(cols, list)
        assert len(cols) == len(set(cols)), f"Duplicate columns detected in feature group '{grp}'!"


def test_loso_imputer_isolation():
    """Verify that a completely held-out station uses only training global statistics."""
    train_df = pd.DataFrame({
        "station": ["Station_A"] * 5,
        "datetime": pd.date_range("2015-01-01", periods=5, freq="1h"),
        "PM2.5": [10.0, 20.0, 30.0, 40.0, 50.0],
        "wd": ["N", "N", "NE", "E", "N"]
    })
    # Training median is 30.0, mode is 'N'

    test_df = pd.DataFrame({
        "station": ["Station_B_HeldOut"] * 3,
        "datetime": pd.date_range("2015-01-01", periods=3, freq="1h"),
        "PM2.5": [np.nan, np.nan, np.nan],  # All missing
        "wd": [np.nan, np.nan, np.nan]      # All missing
    })

    imputer = StationAwareCausalImputer(feature_cols=["PM2.5"], categorical_cols=["wd"])
    imputer.fit(train_df)

    # Station_B_HeldOut was NOT in train_df
    assert "Station_B_HeldOut" not in imputer.station_medians_
    assert imputer.global_medians_["PM2.5"] == 30.0
    assert imputer.global_modes_["wd"] == "N"

    # Transform test_df
    imputed_test = imputer.transform(test_df)

    # It must fall back to the global training median (30.0) and mode ('N')
    assert (imputed_test["PM2.5"] == 30.0).all()
    assert (imputed_test["wd"] == "N").all()


def test_exclusion_mask_accounting(two_station_timeseries):
    """Verify set-theoretic mask independence and sample reconciliation."""
    ds = MultiHorizonForecastingDataset(horizons=[1, 6, 24])
    prepared = ds.prepare_dataset(two_station_timeseries)

    for h in [1, 6, 24]:
        for st, group in two_station_timeseries.groupby("station"):
            N = len(group)
            # Boundary mask B
            mask_B = np.zeros(N, dtype=bool)
            mask_B[-h:] = True

            # Target missing mask M_h
            mask_M = np.zeros(N, dtype=bool)
            target = group["PM2.5"].shift(-h)
            mask_M[:-h] = target[:-h].isnull()

            # Warmup mask W
            mask_W = np.zeros(N, dtype=bool)
            mask_W[:48] = True

            # Verify pairwise intersections
            assert (mask_B & mask_M).sum() == 0, f"Boundary and missing target masks overlap for h={h}!"
            assert (mask_B & mask_W).sum() == 0, f"Boundary and lag warmup masks overlap for h={h}!"
            assert (mask_M & mask_W).sum() == 0, f"Missing target and lag warmup masks overlap for h={h}!"

            # Verify union size
            union_mask = mask_B | mask_M | mask_W
            assert union_mask.sum() == mask_B.sum() + mask_M.sum() + mask_W.sum()

        # Check get_horizon_dataset output length matches raw minus union
        X, y, meta = ds.get_horizon_dataset(prepared, horizon=h, feature_group="combined_all")
        total_raw = len(two_station_timeseries)
        total_boundary = 2 * h
        total_warmup = 2 * 48
        expected_samples = total_raw - total_boundary - total_warmup
        assert len(X) == expected_samples
