"""Chronological purged temporal splitting and leave-one-station-out splitting.

Ensures strict mathematical isolation across evaluation regimes:
1. Temporal Partitioning: enforces a 24-hour purge buffer (embargo) between partitions
   so that multi-horizon targets up to h=24 hours cannot span across boundaries.
2. Spatial Partitioning: isolates held-out stations completely for zero spatial leakage.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class TemporalPurgedSplitter:
    """Chronological train/validation/test splitter with 24-hour boundary purge.

    Partitions:
    - Train: 2013-03-01 00:00:00 to 2015-12-31 23:00:00
    - Purge Buffer 1: 2016-01-01 00:00:00 to 2016-01-01 23:00:00 (24 hours purged)
    - Validation: 2016-01-02 00:00:00 to 2016-09-30 23:00:00
    - Purge Buffer 2: 2016-10-01 00:00:00 to 2016-10-01 23:00:00 (24 hours purged)
    - Test: 2016-10-02 00:00:00 to 2017-02-28 23:00:00
    """

    def __init__(
        self,
        train_end: str = "2015-12-31 23:00:00",
        val_start: str = "2016-01-02 00:00:00",
        val_end: str = "2016-09-30 23:00:00",
        test_start: str = "2016-10-02 00:00:00",
        test_end: str = "2017-02-28 23:00:00",
        datetime_col: str = "datetime",
    ):
        self.train_end = pd.to_datetime(train_end)
        self.val_start = pd.to_datetime(val_start)
        self.val_end = pd.to_datetime(val_end)
        self.test_start = pd.to_datetime(test_start)
        self.test_end = pd.to_datetime(test_end)
        self.datetime_col = datetime_col

        # Validate that purge buffers are at least 24 hours
        buffer1_hours = (self.val_start - self.train_end).total_seconds() / 3600.0
        buffer2_hours = (self.test_start - self.val_end).total_seconds() / 3600.0

        if buffer1_hours < 24.0:
            raise ValueError(f"Train-Val purge buffer must be >= 24h, got {buffer1_hours:.1f}h")
        if buffer2_hours < 24.0:
            raise ValueError(f"Val-Test purge buffer must be >= 24h, got {buffer2_hours:.1f}h")

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into train, validation, and test partitions with boundary purge.

        Args:
            df: DataFrame containing the datetime_col.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if self.datetime_col not in df.columns:
            raise KeyError(f"Datetime column '{self.datetime_col}' not found in DataFrame.")

        dt_series = pd.to_datetime(df[self.datetime_col])

        train_mask = dt_series <= self.train_end
        val_mask = (dt_series >= self.val_start) & (dt_series <= self.val_end)
        test_mask = (dt_series >= self.test_start) & (dt_series <= self.test_end)

        train_df = df[train_mask].copy().reset_index(drop=True)
        val_df = df[val_mask].copy().reset_index(drop=True)
        test_df = df[test_mask].copy().reset_index(drop=True)

        return train_df, val_df, test_df

    def get_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """Compute summary statistics for the temporal splits."""
        train_df, val_df, test_df = self.split(df)
        total_rows = len(df)
        purged_rows = total_rows - (len(train_df) + len(val_df) + len(test_df))

        return {
            "total_rows": total_rows,
            "train_rows": len(train_df),
            "train_pct": round(len(train_df) / total_rows * 100, 2),
            "val_rows": len(val_df),
            "val_pct": round(len(val_df) / total_rows * 100, 2),
            "test_rows": len(test_df),
            "test_pct": round(len(test_df) / total_rows * 100, 2),
            "purged_buffer_rows": purged_rows,
            "purged_pct": round(purged_rows / total_rows * 100, 2),
            "train_window": f"{train_df[self.datetime_col].min()} to {train_df[self.datetime_col].max()}",
            "val_window": f"{val_df[self.datetime_col].min()} to {val_df[self.datetime_col].max()}",
            "test_window": f"{test_df[self.datetime_col].min()} to {test_df[self.datetime_col].max()}",
        }


class StationHoldoutSplitter:
    """Leave-one-station-out cross-station generalisation splitter.

    Isolates one target monitoring station entirely as the test domain,
    while using the remaining 11 stations as training/development domains.
    """

    def __init__(self, station_col: str = "station"):
        self.station_col = station_col

    def split_by_station(
        self, df: pd.DataFrame, held_out_station: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into training stations and a single held-out test station.

        Args:
            df: Combined multi-station DataFrame.
            held_out_station: Identifier of station to hold out.

        Returns:
            Tuple of (train_stations_df, held_out_station_df).
        """
        if self.station_col not in df.columns:
            raise KeyError(f"Station column '{self.station_col}' not found.")

        train_mask = df[self.station_col] != held_out_station
        test_mask = df[self.station_col] == held_out_station

        if not test_mask.any():
            raise ValueError(f"Held out station '{held_out_station}' not present in dataset.")

        train_df = df[train_mask].copy().reset_index(drop=True)
        test_df = df[test_mask].copy().reset_index(drop=True)

        return train_df, test_df
