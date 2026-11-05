"""Multi-horizon forecasting dataset construction and temporal alignment.

Guarantees:
1. Target Integrity: Target labels PM2.5(t+h) for h in {1, 6, 24} hours are constructed
   strictly from RAW un-imputed measurements. Missing targets are dropped, never imputed.
2. Station Isolation: Forward target shifts (t+h) are computed strictly within each station.
3. Information Availability: At prediction time t, X(t) contains only information observed <= t.
4. Clean Horizon Slicing: Convenient extraction of (X, y) matrices for any horizon and feature group.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.features.registry import get_feature_group, FEATURE_GROUPS
from src.features.builder import StationAwareFeatureBuilder
from src.preprocessing.cleaner import StationAwareCausalImputer

HORIZONS = [1, 6, 24]


class MultiHorizonForecastingDataset:
    """Prepares and serves aligned multi-horizon forecasting datasets."""

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        target_col: str = "PM2.5",
        station_col: str = "station",
        datetime_col: str = "datetime",
    ):
        self.horizons = horizons if horizons is not None else HORIZONS
        self.target_col = target_col
        self.station_col = station_col
        self.datetime_col = datetime_col

    def prepare_dataset(
        self,
        raw_df: pd.DataFrame,
        imputer: Optional[StationAwareCausalImputer] = None,
        feature_builder: Optional[StationAwareFeatureBuilder] = None,
        fit_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Construct multi-horizon targets from raw data and build causal features.

        Args:
            raw_df: DataFrame containing raw, un-imputed station records.
            imputer: Optional fitted/unfitted StationAwareCausalImputer.
            feature_builder: Optional StationAwareFeatureBuilder.
            fit_df: Training partition used to fit the imputer if imputer is not already fitted.
                    Guarantees imputer statistics are derived strictly from permitted training data.

        Returns:
            DataFrame with causal features and separate target_h{1,6,24} columns.
        """
        df_work = raw_df.copy()

        # Step 1: Preserve RAW target column for ground truth target construction
        # Targets MUST be derived from raw un-imputed measurements
        raw_target_col = f"{self.target_col}_raw_unimputed"
        df_work[raw_target_col] = df_work[self.target_col]

        # Step 2: Construct multi-horizon targets independently per station
        station_dfs = []
        for station, group in df_work.groupby(self.station_col, sort=False):
            st_group = group.sort_values(self.datetime_col).copy()
            for h in self.horizons:
                # Shift backward by h steps: row at time t receives value from time t+h
                st_group[f"target_h{h}"] = st_group[raw_target_col].shift(-h)
            station_dfs.append(st_group)

        df_work = pd.concat(station_dfs, ignore_index=True)

        # Step 3: Causally impute historical input features (never touches target_h*)
        if imputer is not None:
            if not imputer.is_fitted:
                training_data = fit_df if fit_df is not None else df_work
                imputer.fit(training_data)
            df_work = imputer.transform(df_work)

        # Step 4: Construct station-aware lag, rolling, cyclical, and physical features
        if feature_builder is None:
            feature_builder = StationAwareFeatureBuilder(
                target_col=self.target_col,
                station_col=self.station_col,
                datetime_col=self.datetime_col,
            )

        df_features = feature_builder.build_features(df_work)
        return df_features

    def get_horizon_dataset(
        self,
        df: pd.DataFrame,
        horizon: int,
        feature_group: str = "combined_all",
        drop_initial_lags: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Extract aligned feature matrix X, target vector y, and metadata for a specific horizon.

        Args:
            df: DataFrame produced by prepare_dataset().
            horizon: Forecasting horizon in hours (1, 6, or 24).
            feature_group: Name of feature group (e.g. 'temporal_baseline', 'combined_all').
            drop_initial_lags: Whether to drop initial rows with NaN in lag features (e.g. first 48h).

        Returns:
            Tuple of (X, y, metadata_df) where y is guaranteed non-null.
        """
        target_col = f"target_h{horizon}"
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found. Available targets: {[f'target_h{h}' for h in self.horizons]}")

        # Filter 1: Drop samples where the future target was unobserved
        valid_target_mask = df[target_col].notnull()
        df_valid = df[valid_target_mask].copy()

        # Retrieve feature column names
        feature_cols = get_feature_group(feature_group)

        # Filter 2: Drop initial rows if lag features are NaN (e.g., initial 48h warmup period)
        if drop_initial_lags:
            lag_cols = [c for c in feature_cols if "_lag_" in c]
            if lag_cols:
                valid_lag_mask = df_valid[lag_cols].notnull().all(axis=1)
                df_valid = df_valid[valid_lag_mask]

        X = df_valid[feature_cols].copy()
        y = df_valid[target_col].copy()

        metadata_cols = [self.datetime_col, self.station_col, "year", "month", "day", "hour"]
        available_meta = [c for c in metadata_cols if c in df_valid.columns]
        metadata = df_valid[available_meta].copy()

        return X, y, metadata
