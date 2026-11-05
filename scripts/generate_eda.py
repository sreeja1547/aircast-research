"""Research-grade Exploratory Data Analysis (EDA) script for AirCast Research.

Generates:
1. Full statistical summary JSON saved to artifacts/metrics/eda_statistical_summary.json
2. Eight publication-quality analytical plots saved to artifacts/plots/
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import load_all_stations, get_station_names

# Style configuration for scientific publication
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

OUTPUT_PLOT_DIR = "artifacts/plots"
OUTPUT_METRIC_DIR = "artifacts/metrics"
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_METRIC_DIR, exist_ok=True)


def run_eda():
    print("Loading all station data...")
    df = load_all_stations()
    total_records = len(df)
    print(f"Loaded {total_records} observations across {df['station'].nunique()} stations.")

    eda_metrics = {
        "dataset_total_records": total_records,
        "pollutant_statistics": {},
        "temporal_statistics": {},
        "spatial_statistics": {},
        "meteorological_statistics": {},
        "correlation_matrices": {},
        "missingness_statistics": {},
        "pm_ratio_discrepancy": {},
        "extreme_pollution_statistics": {}
    }

    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    meteo_vars = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

    # -------------------------------------------------------------
    # 1. POLLUTANT DISTRIBUTION STATISTICS & PLOT
    # -------------------------------------------------------------
    print("1. Computing pollutant distribution statistics...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, pol in enumerate(pollutants):
        series = df[pol].dropna()
        skew_val = float(stats.skew(series))
        kurt_val = float(stats.kurtosis(series))
        jb_stat, jb_pval = stats.jarque_bera(series)

        eda_metrics["pollutant_statistics"][pol] = {
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "median": round(float(series.median()), 2),
            "iqr": round(float(series.quantile(0.75) - series.quantile(0.25)), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "skewness": round(skew_val, 3),
            "kurtosis": round(kurt_val, 3),
            "jarque_bera_stat": round(float(jb_stat), 1),
            "jarque_bera_pval": float(jb_pval)
        }

        ax = axes[idx]
        sns.histplot(series, bins=50, kde=True, ax=ax, color="#1f77b4", edgecolor="none", alpha=0.6)
        ax.axvline(series.median(), color="#d62728", linestyle="--", linewidth=1.5, label=f"Median: {series.median():.1f}")
        ax.axvline(series.mean(), color="#2ca02c", linestyle="-", linewidth=1.5, label=f"Mean: {series.mean():.1f}")
        ax.set_title(f"{pol} (Skew: {skew_val:.2f}, Kurt: {kurt_val:.1f})")
        ax.set_xlabel(f"{pol} (ug/m3)")
        ax.legend(fontsize=8)

    plt.suptitle("Pollutant Concentration Distributions & Positive Skewness", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "eda_pollutant_distributions.png"))
    plt.close()

    # -------------------------------------------------------------
    # 2. TEMPORAL & SEASONAL DYNAMICS
    # -------------------------------------------------------------
    print("2. Computing temporal and seasonal dynamics...")
    # Define heating season in Beijing: Nov 15 to Mar 15
    df["is_heating_season"] = df["month"].isin([11, 12, 1, 2, 3])
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["is_weekend"] = df["datetime"].dt.dayofweek >= 5

    diurnal_heating = df[df["is_heating_season"]].groupby("hour")["PM2.5"].mean()
    diurnal_non_heating = df[~df["is_heating_season"]].groupby("hour")["PM2.5"].mean()
    monthly_avg = df.groupby(["year", "month"])["PM2.5"].mean().reset_index()

    eda_metrics["temporal_statistics"] = {
        "heating_season_mean_pm25": round(float(df[df["is_heating_season"]]["PM2.5"].mean()), 2),
        "non_heating_season_mean_pm25": round(float(df[~df["is_heating_season"]]["PM2.5"].mean()), 2),
        "heating_ratio": round(float(df[df["is_heating_season"]]["PM2.5"].mean() / df[~df["is_heating_season"]]["PM2.5"].mean()), 2),
        "weekday_mean_pm25": round(float(df[~df["is_weekend"]]["PM2.5"].mean()), 2),
        "weekend_mean_pm25": round(float(df[df["is_weekend"]]["PM2.5"].mean()), 2),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Diurnal comparison
    axes[0].plot(diurnal_heating.index, diurnal_heating.values, marker="o", color="#d62728", label="Heating (Nov-Mar)")
    axes[0].plot(diurnal_non_heating.index, diurnal_non_heating.values, marker="s", color="#1f77b4", label="Non-Heating (Apr-Oct)")
    axes[0].set_title("Diurnal Cycle: Heating vs. Non-Heating")
    axes[0].set_xlabel("Hour of Day (Local Beijing Time)")
    axes[0].set_ylabel("Mean PM2.5 (ug/m3)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend()

    # Day of week
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_mean = df.groupby("day_of_week")["PM2.5"].mean().reindex(dow_order)
    axes[1].bar(dow_order, dow_mean.values, color="#2b5c8f", alpha=0.85)
    axes[1].set_title("Day-of-Week PM2.5 Concentration")
    axes[1].set_ylabel("Mean PM2.5 (ug/m3)")
    axes[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    # Monthly profile
    month_profile = df.groupby("month")["PM2.5"].agg(["mean", "median"])
    axes[2].plot(month_profile.index, month_profile["mean"], marker="o", color="#d62728", label="Mean")
    axes[2].plot(month_profile.index, month_profile["median"], marker="^", color="#2ca02c", label="Median")
    axes[2].set_title("Annual Cycle by Calendar Month")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("PM2.5 (ug/m3)")
    axes[2].set_xticks(range(1, 13))
    axes[2].legend()

    plt.suptitle("Temporal, Diurnal, and Seasonal Variations of PM2.5", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "eda_temporal_seasonal_patterns.png"))
    plt.close()

    # -------------------------------------------------------------
    # 3. STATION-WISE COMPARISON & SPATIAL GRADIENTS
    # -------------------------------------------------------------
    print("3. Computing station-wise spatial statistics...")
    station_stats = df.groupby("station")["PM2.5"].agg([
        "count", "mean", "std", "median",
        lambda x: x.quantile(0.75) - x.quantile(0.25)
    ]).rename(columns={"<lambda_0>": "iqr"}).sort_values("median")

    for st, row in station_stats.iterrows():
        eda_metrics["spatial_statistics"][st] = {
            "mean": round(float(row["mean"]), 2),
            "std": round(float(row["std"]), 2),
            "median": round(float(row["median"]), 2),
            "iqr": round(float(row["iqr"]), 2)
        }

    plt.figure(figsize=(12, 6))
    order = station_stats.index.tolist()
    sns.boxplot(x="station", y="PM2.5", data=df, order=order, showfliers=False, palette="mako")
    plt.xticks(rotation=45, ha="right")
    plt.title("Spatial Pollution Gradient Across 12 Monitoring Stations (Sorted by Median)")
    plt.xlabel("Monitoring Station")
    plt.ylabel("PM2.5 (ug/m3, Outliers Omitted for Box Visibility)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "eda_station_comparison.png"))
    plt.close()

    # -------------------------------------------------------------
    # 4. METEOROLOGICAL RELATIONSHIPS
    # -------------------------------------------------------------
    print("4. Computing meteorological relationships...")
    # Calculate Dew Point Depression (TEMP - DEWP): smaller depression = higher relative humidity
    df["dew_point_depression"] = df["TEMP"] - df["DEWP"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PM2.5 vs Wind Speed
    wspm_bins = pd.cut(df["WSPM"], bins=np.arange(0, 10, 0.5))
    wspm_pm = df.groupby(wspm_bins, observed=True)["PM2.5"].agg(["mean", "median"])
    bin_centers = [b.mid for b in wspm_pm.index]
    axes[0, 0].plot(bin_centers, wspm_pm["median"], marker="o", color="#1f77b4", label="Median PM2.5")
    axes[0, 0].plot(bin_centers, wspm_pm["mean"], marker="s", color="#d62728", label="Mean PM2.5")
    axes[0, 0].axvline(2.0, color="gray", linestyle=":", label="Dispersion Threshold (2 m/s)")
    axes[0, 0].set_title("Atmospheric Dispersion: PM2.5 vs. Wind Speed")
    axes[0, 0].set_xlabel("Wind Speed (m/s)")
    axes[0, 0].set_ylabel("PM2.5 (ug/m3)")
    axes[0, 0].legend()

    # PM2.5 vs Temperature
    temp_bins = pd.cut(df["TEMP"], bins=np.arange(-20, 42, 4))
    temp_pm = df.groupby(temp_bins, observed=True)["PM2.5"].median()
    t_centers = [b.mid for b in temp_pm.index]
    axes[0, 1].plot(t_centers, temp_pm.values, marker="o", color="#ff7f0e")
    axes[0, 1].set_title("Temperature Inversion vs. Summer Convection")
    axes[0, 1].set_xlabel("Ambient Temperature (°C)")
    axes[0, 1].set_ylabel("Median PM2.5 (ug/m3)")

    # PM2.5 vs Dew Point Depression (RH proxy)
    dp_bins = pd.cut(df["dew_point_depression"], bins=np.arange(0, 35, 2))
    dp_pm = df.groupby(dp_bins, observed=True)["PM2.5"].median()
    dp_centers = [b.mid for b in dp_pm.index]
    axes[1, 0].plot(dp_centers, dp_pm.values, marker="^", color="#2ca02c")
    axes[1, 0].set_title("Moisture & Secondary Aerosols: PM2.5 vs. Dew Point Depression")
    axes[1, 0].set_xlabel("Dew Point Depression: TEMP - DEWP (°C, Small = High RH)")
    axes[1, 0].set_ylabel("Median PM2.5 (ug/m3)")

    # Wind Direction Pollution Rose
    wd_order = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    wd_pm = df.groupby("wd")["PM2.5"].mean().reindex(wd_order)
    axes[1, 1].bar(wd_order, wd_pm.values, color=["#1f77b4" if d in ["NW", "NNW", "N"] else "#d62728" for d in wd_order], alpha=0.8)
    axes[1, 1].set_title("PM2.5 by Wind Direction Sector (Southerly Transport vs Northerly Clean)")
    axes[1, 1].set_xlabel("Wind Direction Sector")
    axes[1, 1].set_ylabel("Mean PM2.5 (ug/m3)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.suptitle("Synoptic Meteorological Drivers of Particulate Pollution", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "eda_meteorological_relationships.png"))
    plt.close()

    # -------------------------------------------------------------
    # 5. CORRELATION ANALYSIS (PEARSON & SPEARMAN)
    # -------------------------------------------------------------
    print("5. Computing linear (Pearson) and non-linear (Spearman) correlation matrices...")
    analysis_cols = pollutants + meteo_vars + ["dew_point_depression"]
    pearson_corr = df[analysis_cols].corr(method="pearson").round(3)
    spearman_corr = df[analysis_cols].corr(method="spearman").round(3)

    eda_metrics["correlation_matrices"]["pearson"] = pearson_corr.to_dict()
    eda_metrics["correlation_matrices"]["spearman"] = spearman_corr.to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("Pearson Linear Correlation Matrix")

    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Spearman Rank Correlation Matrix")

    plt.suptitle("Inter-Pollutant and Meteorological Correlation Structure", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "eda_correlation_matrix.png"))
    plt.close()

    # -------------------------------------------------------------
    # 6. MISSINGNESS PATTERNS & STREAK ANALYSIS
    # -------------------------------------------------------------
    print("6. Analyzing missingness streaks and temporal gaps...")
    # Calculate consecutive missing streak lengths for PM2.5 per station
    streak_records = []
    for st, group in df.groupby("station"):
        s = group["PM2.5"].isnull()
        streaks = (~s).cumsum()[s].value_counts().values
        for l in streaks:
            streak_records.append({"station": st, "streak_length": l})

    streak_df = pd.DataFrame(streak_records)
    eda_metrics["missingness_statistics"] = {
        "pm25_total_missing_records": int(df["PM2.5"].isnull().sum()),
        "pm25_pct_missing": round(float(df["PM2.5"].isnull().sum() / total_records * 100), 4),
        "streak_length_median": float(streak_df["streak_length"].median()) if len(streak_df) > 0 else 0,
        "streak_length_p90": float(streak_df["streak_length"].quantile(0.90)) if len(streak_df) > 0 else 0,
        "streak_length_max": int(streak_df["streak_length"].max()) if len(streak_df) > 0 else 0,
        "pct_streaks_le_3h": round(float((streak_df["streak_length"] <= 3).mean() * 100), 2) if len(streak_df) > 0 else 0,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Missingness percentage by station
    st_missing = df.groupby("station")[pollutants].apply(lambda g: g.isnull().mean() * 100)
    sns.heatmap(st_missing, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[0], cbar_kws={"label": "Missing Rate (%)"})
    axes[0].set_title("Unobserved Rates (%) by Station and Pollutant")
    axes[0].set_ylabel("Monitoring Station")

    # Streak length histogram
    sns.histplot(streak_df["streak_length"], bins=range(1, 40), ax=axes[1], color="#8c564b")
    axes[1].axvline(streak_df["streak_length"].median(), color="red", linestyle="--", label=f"Median: {streak_df['streak_length'].median():.0f}h")
    axes[1].set_title("Distribution of Consecutive Missing Hours (PM2.5 Streaks)")
    axes[1].set_xlabel("Consecutive Missing Duration (Hours)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.suptitle("Missing Data Topology: Station Rates and Gap Durations", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "eda_missingness_patterns.png"))
    plt.close()

    # -------------------------------------------------------------
    # 7. PM2.5 > PM10 RATIO ANOMALY INVESTIGATION
    # -------------------------------------------------------------
    print("7. Investigating PM2.5 > PM10 discrepancy with exact denominator...")
    simultaneous_valid = df["PM2.5"].notnull() & df["PM10"].notnull()
    simultaneous_count = int(simultaneous_valid.sum())
    pm_greater = df[simultaneous_valid & (df["PM2.5"] > df["PM10"])].copy()
    pm_greater_count = len(pm_greater)
    diff = pm_greater["PM2.5"] - pm_greater["PM10"]

    eda_metrics["pm_ratio_discrepancy"] = {
        "valid_simultaneous_pairs_denominator": simultaneous_count,
        "count_pm25_gt_pm10": pm_greater_count,
        "percentage_of_simultaneous_pairs": round(pm_greater_count / simultaneous_count * 100, 6),
        "mean_discrepancy": round(float(diff.mean()), 2),
        "median_discrepancy": round(float(diff.median()), 2),
        "p95_discrepancy": round(float(diff.quantile(0.95)), 2),
        "max_discrepancy": round(float(diff.max()), 2),
        "station_breakdown": {
            st: int((pm_greater["station"] == st).sum()) for st in get_station_names()
        }
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Scatter sample of PM10 vs PM2.5
    sample_df = df[simultaneous_valid].sample(n=10000, random_state=42)
    axes[0].scatter(sample_df["PM10"], sample_df["PM2.5"], alpha=0.15, s=8, color="#1f77b4")
    axes[0].plot([0, 800], [0, 800], color="red", linestyle="--", label="1:1 Equality (PM2.5 = PM10)")
    axes[0].set_title(f"PM2.5 vs PM10 (N=10k Sample, {eda_metrics['pm_ratio_discrepancy']['percentage_of_simultaneous_pairs']:.2f}% Inverted)")
    axes[0].set_xlabel("PM10 (ug/m3)")
    axes[0].set_ylabel("PM2.5 (ug/m3)")
    axes[0].legend()

    # Discrepancy magnitude distribution
    sns.histplot(diff, bins=50, ax=axes[1], color="#e377c2", log_scale=(False, True))
    axes[1].axvline(diff.median(), color="black", linestyle="--", label=f"Median: {diff.median():.1f} ug/m3")
    axes[1].set_title("Discrepancy Magnitude: (PM2.5 - PM10) > 0")
    axes[1].set_xlabel("Discrepancy (ug/m3)")
    axes[1].set_ylabel("Log Frequency")
    axes[1].legend()

    # Discrepancy by station
    st_discrepancy_pct = (pm_greater.groupby("station").size() / df[simultaneous_valid].groupby("station").size() * 100).sort_values()
    axes[2].barh(st_discrepancy_pct.index, st_discrepancy_pct.values, color="#9467bd")
    axes[2].set_title("Percentage of Inverted Pairs by Station")
    axes[2].set_xlabel("Discrepancy Rate (%)")

    plt.suptitle("Investigation of Physical PM2.5 > PM10 Discrepancy", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "eda_pm_ratio_discrepancy.png"))
    plt.close()

    # -------------------------------------------------------------
    # 8. EXTREME POLLUTION EPISODES
    # -------------------------------------------------------------
    print("8. Computing extreme pollution and persistence statistics...")
    # EPA Air Quality Index Breakpoints for PM2.5 (ug/m3)
    categories = {
        "Good (0-12)": (0, 12),
        "Moderate (12.1-35.4)": (12, 35.4),
        "Unhealthy Sensitive (35.5-55.4)": (35.4, 55.4),
        "Unhealthy (55.5-150.4)": (55.4, 150.4),
        "Very Unhealthy (150.5-250.4)": (150.4, 250.4),
        "Hazardous (>250.4)": (250.4, 1000.0)
    }

    cat_counts = {}
    valid_pm25 = df["PM2.5"].dropna()
    for cat, (low, high) in categories.items():
        cnt = int(((valid_pm25 > low) & (valid_pm25 <= high)).sum())
        cat_counts[cat] = {
            "count": cnt,
            "pct": round(cnt / len(valid_pm25) * 100, 2)
        }

    eda_metrics["extreme_pollution_statistics"]["epa_category_breakdown"] = cat_counts
    eda_metrics["extreme_pollution_statistics"]["records_gt_500"] = int((valid_pm25 > 500).sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Category Bar Chart
    cat_names = list(cat_counts.keys())
    cat_pcts = [cat_counts[c]["pct"] for c in cat_names]
    bar_colors = ["#2ca02c", "#ffbb78", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    axes[0].barh(cat_names, cat_pcts, color=bar_colors)
    axes[0].set_title("PM2.5 Health Category Distribution (EPA Breakpoints)")
    axes[0].set_xlabel("Percentage of Observations (%)")

    # Time series of extreme episode: December 2015 "Airpocalypse / Red Alert"
    dec_2015 = df[(df["datetime"] >= "2015-11-25") & (df["datetime"] <= "2015-12-15")]
    daily_dec = dec_2015.groupby(["datetime", "station"])["PM2.5"].mean().unstack()
    axes[1].plot(daily_dec.index, daily_dec.mean(axis=1), color="#d62728", linewidth=1.8, label="Beijing Network Mean")
    axes[1].fill_between(daily_dec.index, daily_dec.min(axis=1), daily_dec.max(axis=1), color="#d62728", alpha=0.2, label="Station Range")
    axes[1].axhline(250.4, color="purple", linestyle="--", label="Hazardous Threshold (250 ug/m3)")
    axes[1].set_title("Historic Red Alert Episode: Nov 25 - Dec 15, 2015")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("PM2.5 (ug/m3)")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Extreme Pollution Distributions & Historic Severe Haze Episodes", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "eda_extreme_pollution_events.png"))
    plt.close()

    # Save metrics JSON
    with open(os.path.join(OUTPUT_METRIC_DIR, "eda_statistical_summary.json"), "w") as f:
        json.dump(eda_metrics, f, indent=2)

    print("EDA execution successfully finished!")
    print(f"Metrics written to {OUTPUT_METRIC_DIR}/eda_statistical_summary.json")
    print(f"8 plots written to {OUTPUT_PLOT_DIR}/")


if __name__ == "__main__":
    run_eda()
