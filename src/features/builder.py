"""Station-aware feature engineering pipeline.

Constructs:
1. Multi-step autoregressive lags (1, 3, 6, 12, 24, 48 hours) independently within each station.
2. Multi-window causal rolling statistics (mean, std, min, max over 3, 6, 12, 24 hours) using past observations only.
3. Physical meteorological derived features (dew point depression, rain indicator, wind vectors).
4. Periodic cyclical temporal encodings (hour and month).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.preprocessing.transformers import CyclicalTransformer, WindVectorTransformer

DEFAULT_LAGS = [1, 3, 6, 12, 24, 48]
DEFAULT_WINDOWS = [3, 6, 12, 24]


class StationAwareFeatureBuilder:
    """Builds lag, rolling, cyclical, and physical features with station isolation.

    Guarantees:
    - Intra-station calculation: all lags and rolling windows are partitioned strictly by station.
      Rows from station A never bleed into station B.
    - Zero future leakage: rolling windows and lags use strictly historical observations up to time t.
    """

    def __init__(
        self,
        target_col: str = "PM2.5",
        station_col: str = "station",
        datetime_col: str = "datetime",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        self.target_col = target_col
        self.station_col = station_col
        self.datetime_col = datetime_col
        self.lags = lags if lags is not None else DEFAULT_LAGS
        self.rolling_windows = rolling_windows if rolling_windows is not None else DEFAULT_WINDOWS

        self.cyclical_transformer = CyclicalTransformer(hour_col="hour", month_col="month")
        self.wind_transformer = WindVectorTransformer(wd_col="wd", wspm_col="WSPM")

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct full feature set on the provided DataFrame.

        Args:
            df: DataFrame containing cleaned/imputed time series.

        Returns:
            DataFrame augmented with engineered features, sorted by station and datetime.
        """
        df_out = df.copy()

        # 1. Cyclical temporal features
        df_out = self.cyclical_transformer.transform(df_out)

        # Additional temporal flags
        dt = pd.to_datetime(df_out[self.datetime_col])
        df_out["day_of_week"] = dt.dt.dayofweek
        df_out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.float32)
        df_out["is_heating_season"] = df_out["month"].isin([11, 12, 1, 2, 3]).astype(np.float32)

        # 2. Meteorological features
        df_out = self.wind_transformer.transform(df_out)
        if "TEMP" in df_out.columns and "DEWP" in df_out.columns:
            df_out["dew_point_depression"] = df_out["TEMP"] - df_out["DEWP"]
        if "RAIN" in df_out.columns:
            df_out["has_rain"] = (df_out["RAIN"] > 0.0).astype(np.float32)

        # 3. Station-aware Lags and Rolling Windows
        station_dfs = []
        for station, group in df_out.groupby(self.station_col, sort=False):
            st_group = group.sort_values(self.datetime_col).copy()

            # Autoregressive lags: shift(k) looks back k steps
            for lag in self.lags:
                st_group[f"{self.target_col}_lag_{lag}"] = st_group[self.target_col].shift(lag)

            # Rolling statistics over past observations up to current time t
            # min_periods=1 ensures that initial points still compute valid statistics
            for w in self.rolling_windows:
                roller = st_group[self.target_col].rolling(window=w, min_periods=1)
                st_group[f"{self.target_col}_roll_mean_{w}"] = roller.mean()
                st_group[f"{self.target_col}_roll_std_{w}"] = roller.std().fillna(0.0)
                st_group[f"{self.target_col}_roll_min_{w}"] = roller.min()
                st_group[f"{self.target_col}_roll_max_{w}"] = roller.max()

            station_dfs.append(st_group)

        result_df = pd.concat(station_dfs, ignore_index=True)
        result_df = result_df.sort_values([self.datetime_col, self.station_col]).reset_index(drop=True)
        return result_df
