"""Experiment 4: Hyperparameter Optimization and Validation Response Surfaces.

Performs systematic grid exploration of gradient boosting hyperparameters
(learning rate, tree depth, stochastic subsampling, feature fraction) strictly
optimizing on the chronologically purged validation partition.
"""

import itertools
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
from src.models import XGBoostForecaster
from src.preprocessing.splitter import TemporalPurgedSplitter


def run_experiment_4(
    cache_path: str = "data/processed/forecasting_features_cache.parquet",
    output_metrics_path: str = "artifacts/metrics/experiment4_hyperparameter_tuning.json",
    output_plot_path: str = "artifacts/plots/hyperparameter_tuning_response.png",
    horizon: int = 24,
) -> Dict[str, Any]:
    print("=" * 80)
    print("AIRCAST RESEARCH: EXPERIMENT 4 — HYPERPARAMETER OPTIMIZATION (h=24h)")
    print("=" * 80)

    Path(output_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading cached features from '{cache_path}'...")
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

    print(f"Purged Partition Samples: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

    param_grid = {
        "learning_rate": [0.03, 0.06, 0.10],
        "max_depth": [5, 7, 9],
        "subsample": [0.75, 0.90],
    }

    grid_combinations = list(itertools.product(
        param_grid["learning_rate"],
        param_grid["max_depth"],
        param_grid["subsample"],
    ))

    print(f"\n[2/4] Evaluating {len(grid_combinations)} hyperparameter configurations on Validation split...")
    tuning_records: List[Dict[str, Any]] = []

    best_val_rmse = float("inf")
    best_config: Dict[str, Any] = {}
    best_model: Any = None

    for idx, (lr, depth, subsample) in enumerate(grid_combinations, 1):
        print(f"---> Config {idx:02d}/{len(grid_combinations):02d}: lr={lr}, max_depth={depth}, subsample={subsample}...", end="", flush=True)

        model = XGBoostForecaster(
            name=f"xgb_tune_{idx}",
            n_estimators=160,
            max_depth=depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=0.8,
            early_stopping_rounds=10,
            n_jobs=-1,
            random_state=42,
        )

        t0 = time.perf_counter()
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        train_time = round(time.perf_counter() - t0, 3)

        val_preds = model.predict(X_val)
        val_metrics = calculate_regression_metrics(y_val, val_preds)

        print(f" Val RMSE: {val_metrics['rmse']} | Val MAE: {val_metrics['mae']} ({train_time}s)")

        rec = {
            "config_id": idx,
            "learning_rate": lr,
            "max_depth": depth,
            "subsample": subsample,
            "train_time_sec": train_time,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
        }
        tuning_records.append(rec)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_config = rec
            best_model = model

    print(f"\n[3/4] Optimal Validation Config Identified: lr={best_config['learning_rate']}, depth={best_config['max_depth']}, subsample={best_config['subsample']}")
    print(f"      Best Val RMSE: {best_val_rmse:.4f} ug/m^3")

    # Final unbiased evaluation of the optimal model on Test partition
    test_preds = best_model.predict(X_test)
    test_metrics = calculate_regression_metrics(y_test, test_preds)
    print(f"      Unbiased Test Evaluation -> MAE: {test_metrics['mae']}, RMSE: {test_metrics['rmse']}, R2: {test_metrics['r2']}")

    summary = {
        "experiment": "Experiment 4: Hyperparameter Optimization",
        "horizon_hours": horizon,
        "n_configurations_evaluated": len(grid_combinations),
        "best_config": best_config,
        "test_evaluation_of_best": test_metrics,
        "all_configurations": tuning_records,
    }

    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot response surfaces
    print(f"\n[4/4] Generating tuning response visualization at '{output_plot_path}'...")
    res_df = pd.DataFrame(tuning_records)
    create_tuning_plot(res_df, output_plot_path)

    print("\n" + "=" * 80)
    print("EXPERIMENT 4 HYPERPARAMETER OPTIMIZATION COMPLETE!")
    print("=" * 80)
    return summary


def create_tuning_plot(df: pd.DataFrame, output_path: str):
    """Generate hyperparameter response curve visualization."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Panel A: Learning Rate vs Val RMSE by Max Depth
    sns.lineplot(
        data=df,
        x="learning_rate",
        y="val_rmse",
        hue="max_depth",
        marker="o",
        markersize=9,
        linewidth=2.5,
        ax=axes[0],
        palette="tab10",
    )
    axes[0].set_title("(a) Validation RMSE vs Learning Rate", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Learning Rate ($\eta$)", fontsize=12)
    axes[0].set_ylabel(r"Validation RMSE ($\mu g/m^3$)", fontsize=12)
    axes[0].legend(title="Max Depth", loc="upper right")

    # Panel B: Subsample vs Val RMSE by Max Depth
    sns.boxplot(
        data=df,
        x="max_depth",
        y="val_rmse",
        hue="subsample",
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("(b) Validation RMSE Distribution by Depth and Subsample", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Maximum Tree Depth", fontsize=12)
    axes[1].set_ylabel(r"Validation RMSE ($\mu g/m^3$)", fontsize=12)
    axes[1].legend(title="Subsample Ratio", loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_experiment_4()
