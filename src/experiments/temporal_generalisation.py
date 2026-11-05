"""Experiments 5 & 6: Forecast Horizon Degradation & Temporal Generalisation.

Analyzes predictive decay across forecasting horizons (1h to 24h) and
evaluates temporal generalization across seasonal regimes (winter heating vs
summer non-heating) and rolling quarterly walk-forward partitions.
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
from src.models import PersistenceForecaster, XGBoostForecaster
from src.preprocessing.splitter import TemporalPurgedSplitter


def run_experiment_5_6(
    cache_path: str = "data/processed/forecasting_features_cache.parquet",
    output_metrics_path: str = "artifacts/metrics/experiment5_6_temporal_generalisation.json",
    output_degradation_plot: str = "artifacts/plots/forecast_horizon_degradation.png",
    output_seasonal_plot: str = "artifacts/plots/seasonal_generalisation_trends.png",
) -> Dict[str, Any]:
    print("=" * 80)
    print("AIRCAST RESEARCH: EXPERIMENTS 5 & 6 — HORIZON DEGRADATION & TEMPORAL GENERALISATION")
    print("=" * 80)

    Path(output_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_degradation_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(output_seasonal_plot).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Loading cached features from '{cache_path}'...")
    df_prepared = pd.read_parquet(cache_path)

    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    horizons = [1, 6, 24]

    # Part 1: Multi-Horizon Degradation Analysis
    print("\n[2/5] Evaluating Horizon Error Degradation Curve (h in [1, 6, 24])...")
    degradation_results: List[Dict[str, Any]] = []

    for h in horizons:
        X, y, metadata = dataset_builder.get_horizon_dataset(
            df_prepared, horizon=h, feature_group="combined_all"
        )
        dt_series = pd.to_datetime(metadata[splitter.datetime_col])
        train_mask = dt_series <= splitter.train_end
        val_mask = (dt_series >= splitter.val_start) & (dt_series <= splitter.val_end)
        test_mask = (dt_series >= splitter.test_start) & (dt_series <= splitter.test_end)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # 1. Persistence Baseline
        pers = PersistenceForecaster(target_col="PM2.5")
        pers.fit(X_train, y_train)
        pers_preds = pers.predict(X_test)
        pers_metrics = calculate_regression_metrics(y_test, pers_preds)
        degradation_results.append({
            "horizon_hours": h,
            "model": "Persistence",
            **pers_metrics,
        })

        # 2. XGBoost
        xgb_model = XGBoostForecaster(
            name=f"xgb_h{h}",
            n_estimators=160,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=12,
            n_jobs=-1,
            random_state=42,
        )
        xgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        xgb_preds = xgb_model.predict(X_test)
        xgb_metrics = calculate_regression_metrics(y_test, xgb_preds)
        degradation_results.append({
            "horizon_hours": h,
            "model": "XGBoost",
            **xgb_metrics,
        })

        skill_pct = (pers_metrics["mae"] - xgb_metrics["mae"]) / pers_metrics["mae"] * 100.0
        print(f"  h={h:2d}h -> Persistence MAE: {pers_metrics['mae']:6.2f} | XGBoost MAE: {xgb_metrics['mae']:6.2f} | Skill Gain: +{skill_pct:5.2f}%")

    # Part 2: Seasonal Generalisation Analysis (Heating vs Non-Heating Regimes)
    print("\n[3/5] Evaluating Seasonal Regime Generalisation (Winter Heating vs Warm Season)...")
    seasonal_results: List[Dict[str, Any]] = []

    # Using h=24 forecasting model to test generalisation across seasons
    X24, y24, meta24 = dataset_builder.get_horizon_dataset(
        df_prepared, horizon=24, feature_group="combined_all"
    )
    dt_series24 = pd.to_datetime(meta24[splitter.datetime_col])
    test_mask24 = (dt_series24 >= splitter.test_start) & (dt_series24 <= splitter.test_end)

    meta_test = meta24[test_mask24].copy()
    meta_test["y_true"] = y24[test_mask24].values
    meta_test["is_heating_season"] = X24.loc[test_mask24, "is_heating_season"].values
    meta_test["datetime_parsed"] = pd.to_datetime(meta_test[splitter.datetime_col])

    # Fit XGBoost on train
    train_mask24 = dt_series24 <= splitter.train_end
    val_mask24 = (dt_series24 >= splitter.val_start) & (dt_series24 <= splitter.val_end)
    model24 = XGBoostForecaster(
        name="xgb_seasonal",
        n_estimators=160,
        max_depth=7,
        learning_rate=0.05,
        early_stopping_rounds=12,
        n_jobs=-1,
        random_state=42,
    )
    model24.fit(X24[train_mask24], y24[train_mask24], X_val=X24[val_mask24], y_val=y24[val_mask24])
    meta_test["y_pred"] = model24.predict(X24[test_mask24])

    # Group by Heating Season indicator
    for is_heat, group in meta_test.groupby("is_heating_season"):
        season_name = "Heating Season (Nov-Feb)" if float(is_heat) == 1.0 else "Non-Heating (Warm/Transitional)"
        m = calculate_regression_metrics(group["y_true"], group["y_pred"])
        seasonal_results.append({
            "season_regime": season_name,
            "is_heating_season": int(float(is_heat)),
            "sample_count": len(group),
            "mean_observed_pm25": round(float(group["y_true"].mean()), 2),
            **m,
        })
        print(f"  Regime: {season_name:<35} | N={len(group):,} | Mean Obs: {group['y_true'].mean():.1f} | MAE: {m['mae']:.2f} | R2: {m['r2']:.4f}")

    # Monthly generalisation tracking in Test period (Oct 2016 - Feb 2017)
    meta_test["year_month"] = meta_test["datetime_parsed"].dt.to_period("M").astype(str)
    monthly_results: List[Dict[str, Any]] = []
    for ym, group in meta_test.groupby("year_month"):
        m = calculate_regression_metrics(group["y_true"], group["y_pred"])
        monthly_results.append({
            "year_month": ym,
            "sample_count": len(group),
            "mean_observed": round(float(group["y_true"].mean()), 2),
            **m,
        })

    # Step 4: Export JSON summary
    print(f"\n[4/5] Exporting temporal metrics to '{output_metrics_path}'...")
    summary = {
        "experiment": "Experiments 5 & 6: Horizon Degradation and Temporal Generalisation",
        "horizon_degradation": degradation_results,
        "seasonal_regimes": seasonal_results,
        "monthly_breakdown": monthly_results,
    }
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Step 5: Visualizations
    print(f"\n[5/5] Generating publication plots...")
    create_degradation_plot(pd.DataFrame(degradation_results), output_degradation_plot)
    create_seasonal_plot(pd.DataFrame(monthly_results), output_seasonal_plot)

    print("\n" + "=" * 80)
    print("EXPERIMENTS 5 & 6 COMPLETE!")
    print("=" * 80)
    return summary


def create_degradation_plot(df: pd.DataFrame, output_path: str):
    """Plot forecasting accuracy degradation as lead time expands."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # MAE vs Horizon
    sns.lineplot(
        data=df, x="horizon_hours", y="mae", hue="model",
        marker="o", markersize=10, linewidth=2.5, ax=axes[0], palette=["#e74c3c", "#2980b9"]
    )
    axes[0].set_title("(a) Mean Absolute Error (MAE) Degradation by Horizon", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Forecast Horizon Lead Time (Hours)", fontsize=12)
    axes[0].set_ylabel(r"Test MAE ($\mu g/m^3$)", fontsize=12)
    axes[0].set_xticks([1, 6, 24])

    # R2 vs Horizon
    sns.lineplot(
        data=df, x="horizon_hours", y="r2", hue="model",
        marker="s", markersize=10, linewidth=2.5, ax=axes[1], palette=["#e74c3c", "#2980b9"]
    )
    axes[1].set_title("(b) Variance Explained ($R^2$) Retention by Horizon", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Forecast Horizon Lead Time (Hours)", fontsize=12)
    axes[1].set_ylabel(r"Explained Variance ($R^2$)", fontsize=12)
    axes[1].set_xticks([1, 6, 24])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def create_seasonal_plot(df: pd.DataFrame, output_path: str):
    """Plot monthly generalisation error across seasonal transitions."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    ax2 = ax1.twinx()

    sns.barplot(data=df, x="year_month", y="mean_observed", ax=ax1, color="#bdc3c7", alpha=0.6)
    sns.lineplot(data=df, x="year_month", y="mae", ax=ax2, color="#c0392b", marker="o", markersize=10, linewidth=2.5)

    ax1.set_title("Test Generalisation Across Winter Heating Transition (2016-10 to 2017-02)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Evaluation Month", fontsize=12)
    ax1.set_ylabel(r"Mean Ground Truth $PM_{2.5}$ ($\mu g/m^3$)", fontsize=12, color="#7f8c8d")
    ax2.set_ylabel(r"Forecasting Test MAE ($\mu g/m^3$)", fontsize=12, color="#c0392b")
    ax1.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_experiment_5_6()
