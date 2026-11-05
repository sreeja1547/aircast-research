"""Feature group registry for controlled feature engineering and ablation experiments.

Defines reproducible feature subsets:
1. temporal_baseline: cyclical time and station identifiers only
2. historical_pollutants: temporal baseline + current pollutant readings
3. lag_features: historical pollutants + multi-step lags (1h to 48h)
4. rolling_features: lag features + multi-window rolling stats (3h to 24h)
5. meteorological_features: temporal + pollutants + meteo + wind vectors
6. combined_all: full multimodal feature representation
"""

from typing import Dict, List

# Core raw column sets
POLLUTANT_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEO_COLS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# Explicit Feature Group Definitions
FEATURE_GROUPS: Dict[str, List[str]] = {
    # 1. Temporal Baseline
    "temporal_baseline": [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "day_of_week", "is_weekend",
        "is_heating_season",
        "station"
    ],

    # 2. Historical Pollutants (instantaneous at time t)
    "historical_pollutants": [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "day_of_week", "is_weekend",
        "is_heating_season",
        "station",
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"
    ],

    # 3. Lag Features (autoregressive PM2.5 memory)
    "lag_features": [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "day_of_week", "is_weekend",
        "is_heating_season",
        "station",
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "PM2.5_lag_1", "PM2.5_lag_3", "PM2.5_lag_6",
        "PM2.5_lag_12", "PM2.5_lag_24", "PM2.5_lag_48"
    ],

    # 4. Rolling Features (short- and medium-term trajectory statistics)
    "rolling_features": [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "day_of_week", "is_weekend",
        "is_heating_season",
        "station",
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "PM2.5_lag_1", "PM2.5_lag_3", "PM2.5_lag_6",
        "PM2.5_lag_12", "PM2.5_lag_24", "PM2.5_lag_48",
        # Rolling Means
        "PM2.5_roll_mean_3", "PM2.5_roll_mean_6", "PM2.5_roll_mean_12", "PM2.5_roll_mean_24",
        # Rolling Std
        "PM2.5_roll_std_3", "PM2.5_roll_std_6", "PM2.5_roll_std_12", "PM2.5_roll_std_24",
        # Rolling Min
        "PM2.5_roll_min_3", "PM2.5_roll_min_6", "PM2.5_roll_min_12", "PM2.5_roll_min_24",
        # Rolling Max
        "PM2.5_roll_max_3", "PM2.5_roll_max_6", "PM2.5_roll_max_12", "PM2.5_roll_max_24"
    ],

    # 5. Meteorological Features
    "meteorological_features": [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "day_of_week", "is_weekend",
        "is_heating_season",
        "station",
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "has_rain", "WSPM",
        "dew_point_depression",
        "wd_sin", "wd_cos", "wind_u", "wind_v"
    ]
}


def get_feature_group(group_name: str) -> List[str]:
    """Retrieve column list for a named feature group."""
    if group_name not in FEATURE_GROUPS and group_name != "combined_all":
        valid = list(FEATURE_GROUPS.keys()) + ["combined_all"]
        raise KeyError(f"Unknown feature group '{group_name}'. Valid groups: {valid}")

    if group_name == "combined_all":
        # Union of all features preserving stable order
        seen = set()
        combined = []
        for grp in ["rolling_features", "meteorological_features"]:
            for col in FEATURE_GROUPS[grp]:
                if col not in seen:
                    seen.add(col)
                    combined.append(col)
        return combined

    return list(FEATURE_GROUPS[group_name])
