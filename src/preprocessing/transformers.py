"""Feature transformation module.

Implements:
1. Cyclical encoding for circular variables (hour, month, wind direction).
2. Physical wind vector decomposition (u and v components).
3. Robust scaling fitted strictly on training data to withstand heavy-pollution outliers.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# 16-point meteorological compass mapping in degrees
WIND_DIR_DEGREE_MAP = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


class WindVectorTransformer:
    """Transforms wind direction and speed into physical meteorological Cartesian components.

    In atmospheric physics:
    - u component: zonal wind (positive = blowing eastward)
    - v component: meridional wind (positive = blowing northward)

    Formula:
        u = -WSPM * sin(theta_rad)
        v = -WSPM * cos(theta_rad)
    """

    def __init__(self, wd_col: str = "wd", wspm_col: str = "WSPM"):
        self.wd_col = wd_col
        self.wspm_col = wspm_col

    def fit(self, df: pd.DataFrame) -> "WindVectorTransformer":
        """Stateless transformer; fit returns self."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply wind vector decomposition."""
        df_out = df.copy()

        # Map compass strings to angles in radians
        rad_series = df_out[self.wd_col].map(WIND_DIR_DEGREE_MAP).apply(
            lambda deg: np.radians(deg) if pd.notnull(deg) else np.nan
        )

        # Wind direction sin and cos
        df_out["wd_sin"] = np.sin(rad_series)
        df_out["wd_cos"] = np.cos(rad_series)

        # Physical Cartesian wind vector components
        if self.wspm_col in df_out.columns:
            wspm = df_out[self.wspm_col].fillna(0.0)
            df_out["wind_u"] = -wspm * np.sin(rad_series)
            df_out["wind_v"] = -wspm * np.cos(rad_series)

        return df_out


class CyclicalTransformer:
    """Encodes periodic temporal cycles (hour of day, month of year) into continuous sine/cosine pairs.

    Ensures continuity across period boundaries (e.g. 23:00 to 00:00, December to January).
    """

    def __init__(self, hour_col: str = "hour", month_col: str = "month"):
        self.hour_col = hour_col
        self.month_col = month_col

    def fit(self, df: pd.DataFrame) -> "CyclicalTransformer":
        """Stateless transformer; fit returns self."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cyclical features."""
        df_out = df.copy()

        if self.hour_col in df_out.columns:
            hour_rad = 2.0 * np.pi * df_out[self.hour_col] / 24.0
            df_out["hour_sin"] = np.sin(hour_rad)
            df_out["hour_cos"] = np.cos(hour_rad)

        if self.month_col in df_out.columns:
            month_rad = 2.0 * np.pi * (df_out[self.month_col] - 1.0) / 12.0
            df_out["month_sin"] = np.sin(month_rad)
            df_out["month_cos"] = np.cos(month_rad)

        return df_out


class RobustFeatureScaler:
    """Robust scaler utilizing median and Interquartile Range (IQR).

    Preserves extreme pollution events without letting extreme values dominate standard deviation.
    Must be fitted strictly on the training partition.
    """

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.medians_: Dict[str, float] = {}
        self.iqrs_: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "RobustFeatureScaler":
        """Compute median and IQR on training split only."""
        for col in self.feature_cols:
            if col in df.columns:
                med = float(df[col].median())
                q25 = float(df[col].quantile(0.25))
                q75 = float(df[col].quantile(0.75))
                iqr = q75 - q25
                if iqr == 0.0 or np.isnan(iqr):
                    iqr = 1.0  # Avoid division by zero
                self.medians_[col] = med
                self.iqrs_[col] = iqr

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using training medians and IQRs."""
        if not self.is_fitted:
            raise RuntimeError("RobustFeatureScaler must be fitted before transform.")

        df_out = df.copy()
        for col in self.feature_cols:
            if col in df_out.columns and col in self.medians_:
                df_out[col] = (df_out[col] - self.medians_[col]) / self.iqrs_[col]

        return df_out
