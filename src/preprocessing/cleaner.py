"""Station-aware causal imputation and cleaning module.

Guarantees:
1. Zero lookahead leakage: missing values are imputed strictly from past observations within the same station.
2. Training isolation: fallback medians/modes are computed strictly on training partitions during fit().
3. Target preservation: target columns (PM2.5 at t+h) are NEVER imputed; samples with missing targets are dropped.
4. Missingness indicators: binary indicator flags are created to inform models of unobserved historical inputs.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class StationAwareCausalImputer:
    """Station-aware causal imputer for multivariate environmental time series.

    Imputes missing historical input measurements within each monitoring station
    using causal forward-fill (past observations only). Any initial missing values
    prior to the first valid observation in the training set are back-filled using
    the training partition's median/mode.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        station_col: str = "station",
        add_indicators: bool = True,
        max_ffill_hours: Optional[int] = 72,
    ):
        """
        Args:
            feature_cols: List of numeric column names to causally impute.
            categorical_cols: List of categorical column names (e.g. 'wd') to causally impute.
            station_col: Column containing station identifiers.
            add_indicators: Whether to append binary columns (feature_name_is_missing).
            max_ffill_hours: Maximum consecutive hours to forward-fill. If missing block exceeds
                             this, falls back to training median/mode. None for unlimited ffill.
        """
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols
        self.station_col = station_col
        self.add_indicators = add_indicators
        self.max_ffill_hours = max_ffill_hours
        self.is_fitted = False
        self.station_medians_: Dict[str, Dict[str, float]] = {}
        self.global_medians_: Dict[str, float] = {}
        self.station_modes_: Dict[str, Dict[str, str]] = {}
        self.global_modes_: Dict[str, str] = {}
        self.imputed_feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "StationAwareCausalImputer":
        """Compute training partition medians and modes per station and globally.

        Must ONLY be called on the training split to prevent data leakage.
        """
        exclude = ["No", "year", "month", "day", "hour", "datetime", self.station_col]

        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c].dtype)]

        if self.categorical_cols is None:
            self.categorical_cols = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c].dtype)]

        # Numeric medians
        self.global_medians_ = df[self.feature_cols].median().to_dict()
        self.station_medians_ = {}
        for station, group in df.groupby(self.station_col):
            self.station_medians_[station] = group[self.feature_cols].median().to_dict()

        # Categorical modes
        self.global_modes_ = {}
        for c in self.categorical_cols:
            mode_series = df[c].dropna().mode()
            self.global_modes_[c] = mode_series.iloc[0] if len(mode_series) > 0 else "N/A"

        self.station_modes_ = {}
        for station, group in df.groupby(self.station_col):
            self.station_modes_[station] = {}
            for c in self.categorical_cols:
                mode_series = group[c].dropna().mode()
                self.station_modes_[station][c] = mode_series.iloc[0] if len(mode_series) > 0 else self.global_modes_.get(c, "N/A")

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply strictly causal, station-aware imputation.

        Args:
            df: DataFrame to impute (train, val, or test partition).

        Returns:
            DataFrame with causally imputed values and optional missingness indicators.
        """
        if not self.is_fitted:
            raise RuntimeError("StationAwareCausalImputer must be fitted on training data before transform.")

        df_out = df.copy()

        # Add missingness indicators before filling
        if self.add_indicators:
            for col in self.feature_cols + self.categorical_cols:
                if col in df_out.columns:
                    df_out[f"{col}_missing"] = df_out[col].isnull().astype(np.float32)

        # Impute station by station
        imputed_groups = []
        for station, group in df_out.groupby(self.station_col, sort=False):
            st_group = group.sort_values("datetime").copy()
            st_medians = self.station_medians_.get(station, self.global_medians_)
            st_modes = self.station_modes_.get(station, self.global_modes_)

            # 1. Numeric imputation
            for col in self.feature_cols:
                if col not in st_group.columns:
                    continue

                if self.max_ffill_hours is not None:
                    st_group[col] = st_group[col].ffill(limit=self.max_ffill_hours)
                else:
                    st_group[col] = st_group[col].ffill()

                fallback = st_medians.get(col, self.global_medians_.get(col, 0.0))
                st_group[col] = st_group[col].fillna(fallback)

            # 2. Categorical imputation
            for col in self.categorical_cols:
                if col not in st_group.columns:
                    continue

                if self.max_ffill_hours is not None:
                    st_group[col] = st_group[col].ffill(limit=self.max_ffill_hours)
                else:
                    st_group[col] = st_group[col].ffill()

                fallback_mode = st_modes.get(col, self.global_modes_.get(col, "N/A"))
                st_group[col] = st_group[col].fillna(fallback_mode)

            imputed_groups.append(st_group)

        result_df = pd.concat(imputed_groups, ignore_index=True).sort_values(
            ["datetime", self.station_col]
        ).reset_index(drop=True)

        return result_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on provided data and transform."""
        return self.fit(df).transform(df)


def drop_unobserved_targets(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Strictly filter out observations with unobserved target values.

    Under no circumstances should the forecast target PM2.5(t+h) be synthetically imputed.
    Any instance where target_col is NaN must be excluded from supervised training/evaluation.

    Args:
        df: DataFrame containing the target column.
        target_col: Name of target column (e.g., 'target_h1', 'target_h6', 'target_h24').

    Returns:
        DataFrame containing only valid target rows.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in DataFrame.")
    valid_mask = df[target_col].notnull()
    return df[valid_mask].copy().reset_index(drop=True)
