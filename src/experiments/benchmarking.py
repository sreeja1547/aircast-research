"""Experiment 1: Systematic Multi-Horizon Algorithm Benchmarking.

Evaluates 7 distinct forecaster architectures across 3 horizons (1h, 6h, 24h)
using the combined_all feature group on the chronologically purged test partition.
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
from src.evaluation.metrics import benchmark_latency, calculate_regression_metrics
from src.models import (
    HistGradientBoostingForecaster,
    LinearRegressionForecaster,
    LSTMForecaster,
    PersistenceForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)
from src.preprocessing.splitter import TemporalPurgedSplitter


def run_experiment_1(
    cache_path: str = "data/processed/forecasting_features_cache.parquet",
    output_metrics_path: str = "artifacts/metrics/experiment1_benchmarking.json",
    output_plot_path: str = "artifacts/plots/benchmarking_model_comparison.png",
    models_dir: str = "artifacts/models",
) -> Dict[str, Any]:
    print("=" * 80)
    print("AIRCAST RESEARCH: EXPERIMENT 1 — MULTI-HORIZON ALGORITHM BENCHMARKING")
    print("=" * 80)

    Path(output_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # Load prepared cache
    print(f"\n[1/4] Loading cached features from '{cache_path}'...")
    df_prepared = pd.read_parquet(cache_path)
    print(f"      Loaded DataFrame with shape: {df_prepared.shape}")

    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    horizons = [1, 6, 24]
    model_configs = [
        ("Persistence", lambda: PersistenceForecaster(name="persistence", target_col="PM2.5")),
        ("Linear Regression", lambda: LinearRegressionForecaster(name="linear_regression", scaler_type="robust")),
        ("Ridge", lambda: RidgeForecaster(name="ridge", alpha=10.0, scaler_type="robust")),
        ("HistGradientBoosting", lambda: HistGradientBoostingForecaster(
            name="hist_gradient_boosting", max_iter=150, max_depth=8, learning_rate=0.08, random_state=42
        )),
        ("Random Forest", lambda: RandomForestForecaster(
            name="random_forest", n_estimators=60, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=42
        )),
        ("XGBoost", lambda: XGBoostForecaster(
            name="xgboost", n_estimators=180, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=15, n_jobs=-1, random_state=42
        )),
        ("LSTM", lambda: LSTMForecaster(
            name="lstm", hidden_dim=64, num_layers=2, dropout=0.2, learning_rate=1e-3,
            batch_size=512, max_epochs=15, patience=3, random_state=42
        )),
    ]

    all_results: List[Dict[str, Any]] = []

    for h in horizons:
        print(f"\n" + "-" * 70)
        print(f"BENCHMARKING FORECASTING HORIZON: h = {h} HOUR(S)")
        print("-" * 70)

        # Extract aligned (X, y, metadata)
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

        print(f"Dataset Partitions (h={h}):")
        print(f"  Train : {len(X_train):,} samples (Purged embargo: 24h)")
        print(f"  Val   : {len(X_val):,} samples (Purged embargo: 24h)")
        print(f"  Test  : {len(X_test):,} samples")

        for model_name, model_fn in model_configs:
            print(f"\n---> Training: {model_name} (h={h}h)...", end="", flush=True)
            model = model_fn()
            t_fit_start = time.perf_counter()

            # Train with validation set for early stopping if supported
            if model_name in ["XGBoost", "LSTM"]:
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            else:
                model.fit(X_train, y_train)

            fit_duration = round(time.perf_counter() - t_fit_start, 3)

            # Generate predictions on Test set
            t_pred_start = time.perf_counter()
            y_pred = model.predict(X_test)
            pred_duration = round(time.perf_counter() - t_pred_start, 3)

            # Evaluate metrics
            reg_metrics = calculate_regression_metrics(y_test, y_pred)

            # Latency benchmark on 500 test samples
            latency_sample = X_test.iloc[:500]
            lat_stats = benchmark_latency(model, latency_sample, n_iterations=30)

            # Save model artifact
            save_name = f"{model.name}_h{h}.joblib"
            save_path = os.path.join(models_dir, save_name)
            model.save(save_path)
            model_size_mb = model.get_model_size_mb(save_path)

            print(f" Done in {fit_duration}s | Test MAE: {reg_metrics['mae']} | RMSE: {reg_metrics['rmse']} | R2: {reg_metrics['r2']}")

            result_entry = {
                "horizon_hours": h,
                "model_name": model_name,
                "model_type": model.model_type,
                "training_time_sec": fit_duration,
                "prediction_time_sec": pred_duration,
                "model_size_mb": model_size_mb,
                "test_samples": len(y_test),
                **reg_metrics,
                "mean_per_sample_latency_ms": lat_stats["mean_per_sample_latency_ms"],
                "throughput_samples_per_sec": lat_stats["throughput_samples_per_sec"],
                "artifact_path": save_path,
            }
            all_results.append(result_entry)

    # Step 3: Save results JSON
    print(f"\n[3/4] Exporting comprehensive metrics to '{output_metrics_path}'...")
    summary_output = {
        "experiment": "Experiment 1: Algorithm Benchmarking",
        "feature_group": "combined_all",
        "n_features": 47,
        "results": all_results,
    }
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)

    # Step 4: Generate Publication Plot
    print(f"\n[4/4] Generating publication-grade comparison plots at '{output_plot_path}'...")
    res_df = pd.DataFrame(all_results)
    create_benchmarking_plots(res_df, output_plot_path)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1 BENCHMARKING COMPLETE!")
    print("=" * 80)
    return summary_output


def create_benchmarking_plots(df: pd.DataFrame, output_path: str):
    """Create high-resolution multi-panel benchmarking comparison visualization."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # Palette
    palette = sns.color_palette("muted", n_colors=df["model_name"].nunique())

    # Panel 1: Test MAE across Horizons
    sns.barplot(
        data=df, x="horizon_hours", y="mae", hue="model_name",
        ax=axes[0, 0], palette="tab10"
    )
    axes[0, 0].set_title("(a) Test Mean Absolute Error (MAE) by Horizon", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Forecasting Horizon (Hours)", fontsize=12)
    axes[0, 0].set_ylabel(r"MAE ($\mu g/m^3$)", fontsize=12)
    axes[0, 0].legend(title="Model", loc="upper left", framealpha=0.9)

    # Panel 2: Test RMSE across Horizons
    sns.barplot(
        data=df, x="horizon_hours", y="rmse", hue="model_name",
        ax=axes[0, 1], palette="tab10"
    )
    axes[0, 1].set_title("(b) Test Root Mean Squared Error (RMSE) by Horizon", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Forecasting Horizon (Hours)", fontsize=12)
    axes[0, 1].set_ylabel(r"RMSE ($\mu g/m^3$)", fontsize=12)
    axes[0, 1].legend(title="Model", loc="upper left", framealpha=0.9)

    # Panel 3: Severe Episode MAE (PM2.5 >= 150)
    sns.barplot(
        data=df, x="horizon_hours", y="extreme_mae", hue="model_name",
        ax=axes[1, 0], palette="tab10"
    )
    axes[1, 0].set_title(r"(c) Hazardous Episode MAE ($PM_{2.5} \geq 150\ \mu g/m^3$)", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Forecasting Horizon (Hours)", fontsize=12)
    axes[1, 0].set_ylabel(r"Extreme Episode MAE ($\mu g/m^3$)", fontsize=12)
    axes[1, 0].legend(title="Model", loc="upper left", framealpha=0.9)

    # Panel 4: Latency vs. Accuracy Tradeoff (h=24)
    h24_df = df[df["horizon_hours"] == 24].copy()
    scatter = sns.scatterplot(
        data=h24_df,
        x="mean_per_sample_latency_ms",
        y="mae",
        hue="model_name",
        s=180,
        ax=axes[1, 1],
        palette="tab10",
        legend=False,
    )
    for _, row in h24_df.iterrows():
        axes[1, 1].annotate(
            row["model_name"],
            (row["mean_per_sample_latency_ms"], row["mae"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=10,
            fontweight="semibold",
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("(d) Inference Latency vs Accuracy Tradeoff ($h=24h$)", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Mean Inference Latency (ms/sample, Log Scale)", fontsize=12)
    axes[1, 1].set_ylabel(r"Test MAE ($\mu g/m^3$)", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_experiment_1()
