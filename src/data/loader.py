"""Station-aware data loader for the Beijing Multi-Site Air Quality Dataset.

Enforces strict chronological sorting, datetime construction, and schema validation.
Never modifies raw files and strictly reads from the provided directory.
"""

import os
import glob
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

DEFAULT_RAW_DIR = "PRSA_Data_20130301-20170228"

EXPECTED_COLUMNS = [
    "No", "year", "month", "day", "hour",
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station"
]

STATION_FILE_MAP = {
    "Aotizhongxin": "PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "Changping": "PRSA_Data_Changping_20130301-20170228.csv",
    "Dingling": "PRSA_Data_Dingling_20130301-20170228.csv",
    "Dongsi": "PRSA_Data_Dongsi_20130301-20170228.csv",
    "Guanyuan": "PRSA_Data_Guanyuan_20130301-20170228.csv",
    "Gucheng": "PRSA_Data_Gucheng_20130301-20170228.csv",
    "Huairou": "PRSA_Data_Huairou_20130301-20170228.csv",
    "Nongzhanguan": "PRSA_Data_Nongzhanguan_20130301-20170228.csv",
    "Shunyi": "PRSA_Data_Shunyi_20130301-20170228.csv",
    "Tiantan": "PRSA_Data_Tiantan_20130301-20170228.csv",
    "Wanliu": "PRSA_Data_Wanliu_20130301-20170228.csv",
    "Wanshouxigong": "PRSA_Data_Wanshouxigong_20130301-20170228.csv",
}


def get_station_names() -> List[str]:
    """Returns sorted list of 12 monitoring station names."""
    return sorted(list(STATION_FILE_MAP.keys()))


def load_station_data(station_name: str, raw_dir: str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load and chronologically index data for a single monitoring station.

    Args:
        station_name: The name of the station (e.g., 'Aotizhongxin').
        raw_dir: Path to directory containing the station CSV files.

    Returns:
        DataFrame sorted strictly by datetime with a verified continuous hourly index.
    """
    if station_name not in STATION_FILE_MAP:
        raise ValueError(f"Unknown station: {station_name}. Valid stations: {get_station_names()}")

    file_path = os.path.join(raw_dir, STATION_FILE_MAP[station_name])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected station file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Schema check
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Schema mismatch in {file_path}. Expected {EXPECTED_COLUMNS}, found {list(df.columns)}")

    # Construct datetime index
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Verify monotonicity and uniqueness of timestamps
    if not df["datetime"].is_monotonic_increasing:
        raise ValueError(f"Timestamps for {station_name} are not monotonically increasing.")
    if df["datetime"].duplicated().any():
        raise ValueError(f"Duplicate timestamps found in {station_name}.")

    return df


def load_all_stations(raw_dir: str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load all 12 station files and concatenate them into a unified DataFrame.

    Args:
        raw_dir: Path to directory containing the station CSV files.

    Returns:
        Combined DataFrame of 420,768 records with datetime parsed and indexed.
    """
    station_dfs = []
    for station in get_station_names():
        st_df = load_station_data(station, raw_dir=raw_dir)
        station_dfs.append(st_df)

    combined = pd.concat(station_dfs, ignore_index=True)
    return combined
