"""Phase 6 / Experiment 7: Leave-One-Station-Out (LOSO) Cross-Domain Generalisation.

Systematically evaluates cross-station spatial transferability across all 12
monitoring stations in the Beijing network. Strictly isolates the held-out
station from all training statistics, feature conditioning, and model fitting.
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
from src.models import XGBoostForecaster
from src.preprocessing.splitter import StationHoldoutSplitter


# Beijing station spatial domain classification and approximate coordinates
STATION_METADATA = {
    "Aotizhongxin": {"domain": "Urban Core", "role": "Olympic Sports Park", "lat": 39.982, "lon": 116.397},
    "Changping": {"domain": "Suburban", "role": "Northern Satellite City", "lat": 40.217, "lon": 116.230},
    "Dingling": {"domain": "Regional Background", "role": "Clean Mountain Reference", "lat": 40.292, "lon": 116.220},
    "Dongsi": {"domain": "Urban Core", "role": "Dongcheng Historic Center", "lat": 39.929, "lon": 116.417},
    "Guanyuan": {"domain": "Urban Core", "role": "Xicheng Residential", "lat": 39.929, "lon": 116.339},
    "Gucheng": {"domain": "Industrial Corridor", "role": "Shijingshan Steel District", "lat": 39.914, "lon": 116.184},
    "Huairou": {"domain": "Suburban", "role": "Northeastern Mountain Valley", "lat": 40.328, "lon": 116.628},
    "Nongzhanguan": {"domain": "Urban Core", "role": "Chaoyang Commercial / Embassy", "lat": 39.937, "lon": 116.461},
    "Shunyi": {"domain": "Suburban", "role": "Airport Transportation Corridor", "lat": 40.127, "lon": 116.655},
    "Tiantan": {"domain": "Urban Core", "role": "Temple of Heaven Park", "lat": 39.886, "lon": 116.407},
    "Wanliu": {"domain": "Urban Core", "role": "Haidian Educational & Tech", "lat": 39.987, "lon": 116.287},
    "Wanshouxigong": {"domain": "Urban Core", "role": "Southwestern Traffic Corridor", "lat": 39.878, "lon": 116.352},
}


def run_experiment_7_loso(
    cache_path: str = "data/processed/forecasting_features_cache.parquet",
    output_metrics_path: str = "artifacts/metrics/experiment7_loso_generalisation.json",
    output_plot_path: str = "artifacts/plots/loso_spatial_generalisation.png",
    horizon: int = 24,
) -> Dict[str, Any]:
    print("=" * 80)
    print("AIRCAST RESEARCH: PHASE 6 / EXPERIMENT 7 — LEAVE-ONE-STATION-OUT (LOSO)")
    print("=" * 80)

    Path(output_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading cached features from '{cache_path}'...")
    df_prepared = pd.read_parquet(cache_path)

    dataset_builder = MultiHorizonForecastingDataset()
    X, y, metadata = dataset_builder.get_horizon_dataset(
        df_prepared, horizon=horizon, feature_group="combined_all"
    )

    stations = sorted(metadata["station"].unique())
    print(f"      Loaded {len(X):,} samples across {len(stations)} stations for h={horizon}h.")

    loso_results: List[Dict[str, Any]] = []

    print(f"\n[2/4] Executing 12-Fold Leave-One-Station-Out Evaluation (h={horizon}h)...")
    for idx, held_out_station in enumerate(stations, 1):
        meta_info = STATION_METADATA.get(held_out_station, {"domain": "Unknown", "role": "Unknown"})
        print(f"\n---> Fold {idx:02d}/12: Holding out [{held_out_station}] ({meta_info['domain']} | {meta_info['role']})...", end="", flush=True)

        is_held_out = metadata["station"] == held_out_station
        train_mask = ~is_held_out
        test_mask = is_held_out

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Chronological validation subset from training stations (last 15% of train for early stopping)
        n_train = len(X_train)
        val_split_idx = int(n_train * 0.85)
        X_tr, y_tr = X_train.iloc[:val_split_idx], y_train.iloc[:val_split_idx]
        X_v, y_v = X_train.iloc[val_split_idx:], y_train.iloc[val_split_idx:]

        model = XGBoostForecaster(
            name=f"xgb_loso_{held_out_station}",
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
        model.fit(X_tr, y_tr, X_val=X_v, y_val=y_v)
        fit_time = round(time.perf_counter() - t0, 2)

        preds = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, preds)

        print(f" Done in {fit_time}s | Test MAE: {metrics['mae']} | RMSE: {metrics['rmse']} | R2: {metrics['r2']}")

        loso_results.append({
            "held_out_station": held_out_station,
            "domain": meta_info["domain"],
            "role": meta_info["role"],
            "lat": meta_info.get("lat"),
            "lon": meta_info.get("lon"),
            "test_samples": len(y_test),
            "training_samples": len(X_train),
            "fit_time_sec": fit_time,
            **metrics,
        })

    # Summary statistics across 12 folds
    loso_df = pd.DataFrame(loso_results)
    mean_mae = float(loso_df["mae"].mean())
    std_mae = float(loso_df["mae"].std())
    mean_rmse = float(loso_df["rmse"].mean())
    mean_r2 = float(loso_df["r2"].mean())

    print("\n" + "-" * 70)
    print(f"12-FOLD LOSO AGGREGATE SUMMARY (h={horizon}h):")
    print(f"  Mean Test MAE : {mean_mae:.2f} +/- {std_mae:.2f} ug/m^3")
    print(f"  Mean Test RMSE: {mean_rmse:.2f} ug/m^3")
    print(f"  Mean Test R^2 : {mean_r2:.4f}")
    print("-" * 70)

    # Step 3: Export metrics JSON
    print(f"\n[3/4] Exporting LOSO metrics to '{output_metrics_path}'...")
    summary = {
        "experiment": "Phase 6 / Experiment 7: Leave-One-Station-Out Generalisation",
        "horizon_hours": horizon,
        "n_stations": len(stations),
        "aggregate": {
            "mean_mae": round(mean_mae, 4),
            "std_mae": round(std_mae, 4),
            "mean_rmse": round(mean_rmse, 4),
            "mean_r2": round(mean_r2, 4),
        },
        "per_station_results": loso_results,
    }
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Step 4: Generate publication figure
    print(f"\n[4/4] Generating publication plot at '{output_plot_path}'...")
    create_loso_plot(loso_df, output_plot_path)

    print("\n" + "=" * 80)
    print("PHASE 6 LEAVE-ONE-STATION-OUT GENERALISATION COMPLETE!")
    print("=" * 80)
    return summary


def create_loso_plot(df: pd.DataFrame, output_path: str):
    """Generate publication-grade LOSO cross-station generalisation figure."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

    # Sort by MAE
    df_sorted = df.sort_values("mae").copy()

    # Panel A: MAE by Held-Out Station colored by Domain
    palette = {"Urban Core": "#e74c3c", "Suburban": "#3498db", "Regional Background": "#2ecc71", "Industrial Corridor": "#f39c12"}
    sns.barplot(
        data=df_sorted,
        x="held_out_station",
        y="mae",
        hue="domain",
        ax=axes[0],
        palette=palette,
        dodge=False,
    )
    axes[0].axhline(df["mae"].mean(), color="black", linestyle="--", linewidth=1.5, label=f"Network Mean ({df['mae'].mean():.1f} $\mu g/m^3$)")
    axes[0].set_title("(a) Out-of-Domain Generalisation Error (Test MAE)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Held-Out Monitoring Station", fontsize=12)
    axes[0].set_ylabel(r"Test MAE ($\mu g/m^3$)", fontsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=40, ha="right")
    axes[0].legend(title="Station Domain", loc="upper left")

    # Panel B: Explained Variance (R^2) by Held-Out Station
    sns.barplot(
        data=df_sorted,
        x="held_out_station",
        y="r2",
        hue="domain",
        ax=axes[1],
        palette=palette,
        dodge=False,
    )
    axes[1].axhline(df["r2"].mean(), color="black", linestyle="--", linewidth=1.5, label=f"Network Mean ($R^2={df['r2'].mean():.3f}$)")
    axes[1].set_title("(b) Spatial Transfer Explained Variance ($R^2$)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Held-Out Monitoring Station", fontsize=12)
    axes[1].set_ylabel(r"Test Explained Variance ($R^2$)", fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=40, ha="right")
    axes[1].legend(title="Station Domain", loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_experiment_7_loso()
