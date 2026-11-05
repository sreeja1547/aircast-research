"""Robustness evaluation, distribution shift, and stress testing.

Evaluates model resilience under sensor noise injection, sensor dropout / telemetry outage,
and extreme hazardous pollution episodes (PM2.5 > 150 and > 250 ug/m^3).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.evaluation.metrics import calculate_regression_metrics
from src.models import XGBoostForecaster
from src.preprocessing.splitter import TemporalPurgedSplitter


def run_robustness_evaluation(
    cache_path: str = "data/processed/forecasting_features_cache.parquet",
    output_metrics_path: str = "artifacts/metrics/experiment8_robustness.json",
    output_plot_path: str = "artifacts/plots/robustness_stress_testing.png",
    horizon: int = 24,
) -> Dict[str, Any]:
    print("=" * 80)
    print("AIRCAST RESEARCH: PHASE 8 / EXPERIMENT 8 — ROBUSTNESS & STRESS TESTING")
    print("=" * 80)

    Path(output_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Loading cached features from '{cache_path}'...")
    df_prepared = pd.read_parquet(cache_path)

    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    X, y, metadata = dataset_builder.get_horizon_dataset(
        df_prepared, horizon=horizon, feature_group="combined_all"
    )

    dt_series = pd.to_datetime(metadata[splitter.datetime_col])
    train_mask = dt_series <= splitter.train_end
    val_mask = (dt_series >= splitter.val_start) & (dt_series <= splitter.val_end)
    test_mask = (dt_series >= splitter.test_start) & (dt_series <= splitter.test_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Fit canonical model
    print(f"\n[2/5] Training reference model on clean features (h={horizon}h)...")
    model = XGBoostForecaster(
        name="xgb_robustness",
        n_estimators=160,
        max_depth=7,
        learning_rate=0.05,
        early_stopping_rounds=12,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    clean_preds = model.predict(X_test)
    baseline_metrics = calculate_regression_metrics(y_test, clean_preds)
    print(f"      Clean Baseline Test MAE: {baseline_metrics['mae']:.2f} | RMSE: {baseline_metrics['rmse']:.2f}")

    # Stress Test 1: Sensor Measurement Noise Injection
    print("\n[3/5] Stress Test 1: Sensor Measurement Noise Injection (Gaussian Noise)...")
    noise_levels = [0.05, 0.15, 0.25]
    noise_results: List[Dict[str, Any]] = []

    np.random.seed(42)
    feature_stds = X_train.std(numeric_only=True)

    for sigma in noise_levels:
        X_test_noisy = X_test.copy()
        for col in feature_stds.index:
            if col in X_test_noisy.columns:
                noise = np.random.normal(0, sigma * feature_stds[col], size=len(X_test))
                X_test_noisy[col] = X_test_noisy[col] + noise

        noisy_preds = model.predict(X_test_noisy)
        m = calculate_regression_metrics(y_test, noisy_preds)
        delta_mae = m["mae"] - baseline_metrics["mae"]
        pct_degradation = (delta_mae / baseline_metrics["mae"]) * 100.0

        print(f"  Noise sigma={sigma*100:4.1f}% -> MAE: {m['mae']:.2f} (+{delta_mae:.2f} ug/m^3, +{pct_degradation:.1f}%) | R2: {m['r2']:.4f}")
        noise_results.append({
            "noise_std_ratio": sigma,
            "pct_degradation": round(pct_degradation, 2),
            **m,
        })

    # Stress Test 2: Sensor Telemetry Outages (Zeroed / Dropped Readings)
    print("\n[4/5] Stress Test 2: Sensor Telemetry Outage Simulation...")
    outage_scenarios = [
        ("No Outage (Baseline)", []),
        ("CO Sensor Outage", ["CO"]),
        ("SO2 + NO2 Sensor Outage", ["SO2", "NO2"]),
        ("Complete Secondary Precursor Outage (SO2, NO2, CO, O3)", ["SO2", "NO2", "CO", "O3"]),
        ("Wind Sensor Outage (WSPM, u, v, wd)", ["WSPM", "wind_u", "wind_v", "wd_sin", "wd_cos"]),
    ]

    outage_results: List[Dict[str, Any]] = []
    for scenario_name, dropped_features in outage_scenarios:
        X_test_outage = X_test.copy()
        for f in dropped_features:
            if f in X_test_outage.columns:
                # Replace with training median fallback
                X_test_outage[f] = float(X_train[f].median())

        outage_preds = model.predict(X_test_outage)
        m = calculate_regression_metrics(y_test, outage_preds)
        delta_mae = m["mae"] - baseline_metrics["mae"]

        print(f"  Scenario: {scenario_name:<55} -> MAE: {m['mae']:.2f} (+{delta_mae:.2f}) | R2: {m['r2']:.4f}")
        outage_results.append({
            "scenario": scenario_name,
            "dropped_features": dropped_features,
            "delta_mae": round(delta_mae, 2),
            **m,
        })

    # Stress Test 3: Hazardous Air Quality Episode Stratification
    print("\n[5/5] Stress Test 3: AQI Severity Stratification...")
    strata = [
        ("Good to Moderate (0 - 75 ug/m^3)", 0.0, 75.0),
        ("Unhealthy for Sensitive Groups (75 - 115 ug/m^3)", 75.0, 115.0),
        ("Unhealthy (115 - 150 ug/m^3)", 115.0, 150.0),
        ("Very Unhealthy (150 - 250 ug/m^3)", 150.0, 250.0),
        ("Severe / Hazardous (> 250 ug/m^3)", 250.0, 1000.0),
    ]

    strata_results: List[Dict[str, Any]] = []
    for name, low, high in strata:
        mask = (y_test >= low) & (y_test < high)
        if np.sum(mask) > 0:
            m = calculate_regression_metrics(y_test[mask], clean_preds[mask])
            print(f"  {name:<50} -> N={len(y_test[mask]):,} | MAE: {m['mae']:.2f} | RMSE: {m['rmse']:.2f}")
            strata_results.append({
                "category": name,
                "range_ug_m3": f"[{low}, {high})",
                "sample_count": int(np.sum(mask)),
                **m,
            })

    summary = {
        "experiment": "Phase 8: Robustness Evaluation and Stress Testing",
        "horizon_hours": horizon,
        "clean_baseline": baseline_metrics,
        "noise_stress_test": noise_results,
        "outage_stress_test": outage_results,
        "severity_stratification": strata_results,
    }

    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate visual figure
    create_robustness_plot(
        pd.DataFrame(noise_results),
        pd.DataFrame(outage_results),
        pd.DataFrame(strata_results),
        output_plot_path,
    )

    print("\n" + "=" * 80)
    print("PHASE 8 ROBUSTNESS EVALUATION COMPLETE!")
    print("=" * 80)
    return summary


def create_robustness_plot(
    df_noise: pd.DataFrame,
    df_outage: pd.DataFrame,
    df_strata: pd.DataFrame,
    output_path: str,
):
    """Generate publication figure for robustness and stress testing."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), dpi=300)

    # Panel A: Noise degradation
    sns.lineplot(
        data=df_noise, x="noise_std_ratio", y="mae",
        marker="o", markersize=10, linewidth=2.5, color="#e74c3c", ax=axes[0]
    )
    axes[0].set_title("(a) Degradation Under Feature Noise", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(r"Noise Standard Deviation ($\sigma / \sigma_X$)", fontsize=11)
    axes[0].set_ylabel(r"Test MAE ($\mu g/m^3$)", fontsize=11)

    # Panel B: Outage Scenarios
    sns.barplot(
        data=df_outage, x="scenario", y="mae",
        palette="Blues_r", ax=axes[1]
    )
    axes[1].set_title("(b) Performance Under Sensor Outages", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Sensor Failure Scenario", fontsize=11)
    axes[1].set_ylabel(r"Test MAE ($\mu g/m^3$)", fontsize=11)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right")

    # Panel C: Error by Severity
    sns.barplot(
        data=df_strata, x="category", y="mae",
        palette="Reds", ax=axes[2]
    )
    axes[2].set_title("(c) Error Scaling by Pollution Severity", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Air Quality Severity Tier", fontsize=11)
    axes[2].set_ylabel(r"Test MAE ($\mu g/m^3$)", fontsize=11)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=35, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_robustness_evaluation()
