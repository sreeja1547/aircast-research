"""Experiment 2 & 3: Feature Engineering Ablation & Marginal Predictive Power Analysis.

Evaluates the predictive utility of each registered feature group (temporal,
historical pollutants, autoregressive lags, rolling trajectory statistics,
meteorological physical vectors, and combined multimodal representation) and
conducts systematic leave-one-group-out ablation across multi-step forecasting horizons.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.evaluation.metrics import calculate_regression_metrics
from src.features.registry import FEATURE_GROUPS, get_feature_group
from src.models import XGBoostForecaster
from src.preprocessing.splitter import TemporalPurgedSplitter


def get_ablation_feature_groups() -> Dict[str, List[str]]:
    """Define standard feature groups and leave-one-group-out ablation subsets."""
    groups: Dict[str, List[str]] = {}

    # 1. Standard groups from registry
    for name in [
        "temporal_baseline",
        "historical_pollutants",
        "lag_features",
        "rolling_features",
        "meteorological_features",
        "combined_all",
    ]:
        groups[name] = get_feature_group(name)

    combined_cols = get_feature_group("combined_all")

    # 2. Leave-one-group-out subsets
    meteo_unique = [
        "TEMP", "PRES", "DEWP", "RAIN", "has_rain", "WSPM",
        "dew_point_depression", "wd_sin", "wd_cos", "wind_u", "wind_v"
    ]
    groups["ablation_minus_meteo"] = [c for c in combined_cols if c not in meteo_unique]

    rolling_unique = [c for c in combined_cols if "_roll_" in c]
    groups["ablation_minus_rolling"] = [c for c in combined_cols if c not in rolling_unique]

    lag_unique = [c for c in combined_cols if "_lag_" in c]
    groups["ablation_minus_lags"] = [c for c in combined_cols if c not in lag_unique]

    temporal_unique = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_of_week", "is_weekend", "is_heating_season"
    ]
    groups["ablation_minus_temporal"] = [c for c in combined_cols if c not in temporal_unique]

    return groups


def run_experiment_2_3(
    cache_path: str = "data/processed/forecasting_features_cache.parquet",
    output_metrics_path: str = "artifacts/metrics/experiment2_3_ablation.json",
    output_plot_path: str = "artifacts/plots/feature_ablation_comparison.png",
) -> Dict[str, Any]:
    print("=" * 80)
    print("AIRCAST RESEARCH: EXPERIMENTS 2 & 3 — FEATURE ENGINEERING & ABLATION STUDY")
    print("=" * 80)

    Path(output_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading cached features from '{cache_path}'...")
    df_prepared = pd.read_parquet(cache_path)
    print(f"      Loaded DataFrame with shape: {df_prepared.shape}")

    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()
    ablation_groups = get_ablation_feature_groups()

    horizons = [1, 6, 24]
    all_results: List[Dict[str, Any]] = []

    for h in horizons:
        print(f"\n" + "-" * 70)
        print(f"EVALUATING FEATURE GROUPS FOR HORIZON: h = {h} HOUR(S)")
        print("-" * 70)

        # Base dataset with target_h
        target_col = f"target_h{h}"
        valid_mask = df_prepared[target_col].notnull()
        df_valid = df_prepared[valid_mask].copy()

        # Drop initial 48h lag rows so all feature sets evaluate on identical samples
        lag_cols_all = [c for c in df_valid.columns if "_lag_" in c]
        if lag_cols_all:
            valid_lag_mask = df_valid[lag_cols_all].notnull().all(axis=1)
            df_valid = df_valid[valid_lag_mask]

        dt_series = pd.to_datetime(df_valid[splitter.datetime_col])
        train_mask = dt_series <= splitter.train_end
        val_mask = (dt_series >= splitter.val_start) & (dt_series <= splitter.val_end)
        test_mask = (dt_series >= splitter.test_start) & (dt_series <= splitter.test_end)

        y_train = df_valid.loc[train_mask, target_col]
        y_val = df_valid.loc[val_mask, target_col]
        y_test = df_valid.loc[test_mask, target_col]

        for group_name, cols in ablation_groups.items():
            print(f"---> Horizon h={h}h | Feature Group: {group_name:<25} ({len(cols)} cols)...", end="", flush=True)

            X_train = df_valid.loc[train_mask, cols]
            X_val = df_valid.loc[val_mask, cols]
            X_test = df_valid.loc[test_mask, cols]

            model = XGBoostForecaster(
                name=f"xgb_{group_name}",
                n_estimators=150,
                max_depth=7,
                learning_rate=0.06,
                subsample=0.8,
                colsample_bytree=0.8,
                early_stopping_rounds=10,
                n_jobs=-1,
                random_state=42,
            )

            t0 = time.perf_counter()
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            fit_time = round(time.perf_counter() - t0, 3)

            y_pred = model.predict(X_test)
            metrics = calculate_regression_metrics(y_test, y_pred)

            print(f" Done in {fit_time}s | MAE: {metrics['mae']} | RMSE: {metrics['rmse']} | R2: {metrics['r2']}")

            all_results.append({
                "horizon_hours": h,
                "feature_group": group_name,
                "n_features": len(cols),
                "training_time_sec": fit_time,
                **metrics,
            })

    # Step 3: Export metrics JSON
    print(f"\n[3/4] Exporting ablation metrics to '{output_metrics_path}'...")
    summary = {
        "experiment": "Experiments 2 & 3: Feature Engineering and Ablation Study",
        "model": "XGBoost",
        "results": all_results,
    }
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Step 4: Generate visual report
    print(f"\n[4/4] Generating visualization at '{output_plot_path}'...")
    res_df = pd.DataFrame(all_results)
    create_ablation_plots(res_df, output_plot_path)

    print("\n" + "=" * 80)
    print("EXPERIMENTS 2 & 3 ABLATION STUDY COMPLETE!")
    print("=" * 80)
    return summary


def create_ablation_plots(df: pd.DataFrame, output_path: str):
    """Create publication-grade ablation visualization comparing feature groups."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

    # Part A: Standard Feature Groups Progression
    std_groups = [
        "temporal_baseline",
        "historical_pollutants",
        "lag_features",
        "rolling_features",
        "meteorological_features",
        "combined_all",
    ]
    std_df = df[df["feature_group"].isin(std_groups)].copy()

    sns.barplot(
        data=std_df,
        x="feature_group",
        y="mae",
        hue="horizon_hours",
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title("(a) Performance by Feature Group Hierarchy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Feature Group", fontsize=12)
    axes[0].set_ylabel(r"Test MAE ($\mu g/m^3$)", fontsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=35, ha="right")
    axes[0].legend(title="Horizon (Hours)", loc="upper right")

    # Part B: Marginal Impact of Ablated Components (Delta MAE relative to combined_all)
    ablation_subsets = [
        "ablation_minus_meteo",
        "ablation_minus_rolling",
        "ablation_minus_lags",
        "ablation_minus_temporal",
        "combined_all",
    ]
    abl_df = df[df["feature_group"].isin(ablation_subsets)].copy()

    sns.barplot(
        data=abl_df,
        x="feature_group",
        y="mae",
        hue="horizon_hours",
        ax=axes[1],
        palette="magma",
    )
    axes[1].set_title("(b) Leave-One-Group-Out Ablation (Marginal Sensitivity)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Ablated Feature Subset", fontsize=12)
    axes[1].set_ylabel(r"Test MAE ($\mu g/m^3$)", fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right")
    axes[1].legend(title="Horizon (Hours)", loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_experiment_2_3()
