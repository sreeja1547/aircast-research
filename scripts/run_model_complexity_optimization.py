"""Stage 3: Model Complexity Optimization for AirCast Research.

Evaluates trade-offs between model capacity, training time, memory/disk footprint,
and predictive performance across:
1. Random Forest (baseline vs pruned vs compact).
2. XGBoost (baseline vs pruned vs fast).
3. PyTorch LSTM (baseline 2-layer vs compact single-layer).
4. Linear / Ridge (ultra-low-complexity reference).

Exports:
- artifacts/metrics/model_complexity_optimization.json
- artifacts/plots/model_accuracy_vs_complexity.png
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.evaluation.metrics import calculate_regression_metrics, benchmark_latency
from src.models import (
    LinearRegressionForecaster,
    LSTMForecaster,
    RidgeForecaster,
)
from src.preprocessing.splitter import TemporalPurgedSplitter


def load_baseline_metrics(horizon: int) -> Dict[str, Dict[str, Any]]:
    """Retrieve pre-measured empirical baseline metrics from Stage 1 to avoid redundant re-fitting."""
    baseline_path = Path("artifacts/metrics/efficiency_baseline.json")
    if not baseline_path.exists():
        return {}
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    baseline_map = {}
    for r in data.get("results", []):
        if r.get("horizon_hours") == horizon:
            mname = r["model_name"]
            perf = r["predictive_performance"]
            baseline_map[mname] = {
                "model_name": f"{mname}_Baseline",
                "training_time_sec": r.get("training_time_sec", 0.0),
                "artifact_size_mb": r.get("artifact_size_mb", 0.0),
                "val_mae": None,
                "val_rmse": None,
                "val_r2": None,
                "test_mae": round(perf["mae"], 3),
                "test_rmse": round(perf["rmse"], 3),
                "test_r2": round(perf["r2"], 4),
                "mean_latency_ms": round(r["latency_benchmarks"]["batch_512"]["per_sample_ms"], 5),
                "throughput_samples_per_sec": round(r["throughput_samples_per_sec"], 1),
            }
    return baseline_map


def profile_model_efficiency(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    temp_save_path: str = "artifacts/models/temp_complexity_test.joblib",
) -> Dict[str, Any]:
    """Train model, record training time, artifact size, validation & test accuracy, and latency."""
    t0_fit = time.perf_counter()
    if hasattr(model, "fit"):
        if isinstance(model, xgb.XGBRegressor):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif isinstance(model, LSTMForecaster):
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)
    fit_duration_sec = round(time.perf_counter() - t0_fit, 3)

    # Predictions
    val_preds = model.predict(X_val)
    val_metrics = calculate_regression_metrics(y_val, val_preds)

    test_preds = model.predict(X_test)
    test_metrics = calculate_regression_metrics(y_test, test_preds)

    # Model save size
    if hasattr(model, "save"):
        model.save(temp_save_path)
    else:
        joblib.dump(model, temp_save_path)
    size_mb = round(os.path.getsize(temp_save_path) / (1024 * 1024), 3)

    # Latency benchmark
    lat_stats = benchmark_latency(model, X_test.iloc[:500], n_iterations=20)

    return {
        "model_name": model_name,
        "training_time_sec": fit_duration_sec,
        "artifact_size_mb": size_mb,
        "val_mae": round(val_metrics["mae"], 3),
        "val_rmse": round(val_metrics["rmse"], 3),
        "val_r2": round(val_metrics["r2"], 4),
        "test_mae": round(test_metrics["mae"], 3),
        "test_rmse": round(test_metrics["rmse"], 3),
        "test_r2": round(test_metrics["r2"], 4),
        "mean_latency_ms": lat_stats["mean_per_sample_latency_ms"],
        "throughput_samples_per_sec": lat_stats["throughput_samples_per_sec"],
    }


def main():
    print("=" * 80)
    print("STAGE 3: MODEL COMPLEXITY OPTIMIZATION & PRUNING EXPERIMENTS")
    print("=" * 80)

    cache_path = "data/processed/forecasting_features_cache.parquet"
    print(f"\n[1/4] Loading cached features from '{cache_path}'...")
    df_prepared = pd.read_parquet(cache_path)

    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    horizons = [1, 6, 24]
    complexity_results = {}
    temp_path = "artifacts/models/temp_complexity_eval.joblib"

    for h in horizons:
        print(f"\n" + "-" * 70)
        print(f"COMPLEXITY OPTIMIZATION FOR HORIZON: h = {h} HOUR(S)")
        print("-" * 70)

        baselines = load_baseline_metrics(h)
        horizon_runs = []

        X_full, y_full, meta = dataset_builder.get_horizon_dataset(
            df_prepared, horizon=h, feature_group="combined_all"
        )
        # Ensure numerical encoding
        for col in X_full.columns:
            if not pd.api.types.is_numeric_dtype(X_full[col].dtype):
                uvals = sorted([str(v) for v in X_full[col].dropna().unique()])
                smap = {v: float(i) for i, v in enumerate(uvals)}
                X_full[col] = X_full[col].astype(str).map(smap).fillna(-1.0).astype(np.float32)

        dt_series = pd.to_datetime(meta[splitter.datetime_col])
        train_mask = dt_series <= splitter.train_end
        val_mask = (dt_series >= splitter.val_start) & (dt_series <= splitter.val_end)
        test_mask = (dt_series >= splitter.test_start) & (dt_series <= splitter.test_end)

        X_train, y_train = X_full[train_mask], y_full[train_mask]
        X_val, y_val = X_full[val_mask], y_full[val_mask]
        X_test, y_test = X_full[test_mask], y_full[test_mask]

        # 1. Random Forest: Baseline from Stage 1 + Pruned + Compact
        if "Random Forest" in baselines:
            rf_base = baselines["Random Forest"]
            rf_base["model_name"] = "RandomForest_Baseline (60 trees, depth 14)"
            horizon_runs.append(rf_base)
            print(f"  [Ref] {rf_base['model_name']} | MAE: {rf_base['test_mae']} | Size: {rf_base['artifact_size_mb']}MB | Throughput: {rf_base['throughput_samples_per_sec']:.0f} s/s")

        pruned_rf = RandomForestRegressor(n_estimators=35, max_depth=10, min_samples_leaf=8, n_jobs=-1, random_state=42)
        print("  ---> Benchmarking RandomForest_Pruned (35 trees, depth 10, leaf 8)...", end="", flush=True)
        res_rf_pruned = profile_model_efficiency(pruned_rf, X_train, y_train, X_val, y_val, X_test, y_test, "RandomForest_Pruned (35 trees, depth 10, leaf 8)", temp_path)
        horizon_runs.append(res_rf_pruned)
        print(f" Done! Test MAE: {res_rf_pruned['test_mae']} | Size: {res_rf_pruned['artifact_size_mb']}MB | Throughput: {res_rf_pruned['throughput_samples_per_sec']:.0f} s/s | Train: {res_rf_pruned['training_time_sec']}s")

        compact_rf = RandomForestRegressor(n_estimators=20, max_depth=8, min_samples_leaf=12, n_jobs=-1, random_state=42)
        print("  ---> Benchmarking RandomForest_Compact (20 trees, depth 8, leaf 12)...", end="", flush=True)
        res_rf_compact = profile_model_efficiency(compact_rf, X_train, y_train, X_val, y_val, X_test, y_test, "RandomForest_Compact (20 trees, depth 8, leaf 12)", temp_path)
        horizon_runs.append(res_rf_compact)
        print(f" Done! Test MAE: {res_rf_compact['test_mae']} | Size: {res_rf_compact['artifact_size_mb']}MB | Throughput: {res_rf_compact['throughput_samples_per_sec']:.0f} s/s | Train: {res_rf_compact['training_time_sec']}s")

        # 2. XGBoost: Baseline + Pruned + Fast
        if "XGBoost" in baselines:
            xgb_base = baselines["XGBoost"]
            xgb_base["model_name"] = "XGBoost_Baseline (180 trees, depth 7, lr 0.05)"
            horizon_runs.append(xgb_base)
            print(f"  [Ref] {xgb_base['model_name']} | MAE: {xgb_base['test_mae']} | Size: {xgb_base['artifact_size_mb']}MB | Throughput: {xgb_base['throughput_samples_per_sec']:.0f} s/s")

        pruned_xgb = xgb.XGBRegressor(n_estimators=90, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                                      tree_method="hist", early_stopping_rounds=12, n_jobs=-1, random_state=42, objective="reg:squarederror")
        print("  ---> Benchmarking XGBoost_Pruned (90 trees, depth 6, lr 0.08)...", end="", flush=True)
        res_xgb_pruned = profile_model_efficiency(pruned_xgb, X_train, y_train, X_val, y_val, X_test, y_test, "XGBoost_Pruned (90 trees, depth 6, lr 0.08)", temp_path)
        horizon_runs.append(res_xgb_pruned)
        print(f" Done! Test MAE: {res_xgb_pruned['test_mae']} | Size: {res_xgb_pruned['artifact_size_mb']}MB | Throughput: {res_xgb_pruned['throughput_samples_per_sec']:.0f} s/s | Train: {res_xgb_pruned['training_time_sec']}s")

        fast_xgb = xgb.XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.12, subsample=0.8, colsample_bytree=0.8,
                                    tree_method="hist", early_stopping_rounds=10, n_jobs=-1, random_state=42, objective="reg:squarederror")
        print("  ---> Benchmarking XGBoost_Fast (50 trees, depth 5, lr 0.12)...", end="", flush=True)
        res_xgb_fast = profile_model_efficiency(fast_xgb, X_train, y_train, X_val, y_val, X_test, y_test, "XGBoost_Fast (50 trees, depth 5, lr 0.12)", temp_path)
        horizon_runs.append(res_xgb_fast)
        print(f" Done! Test MAE: {res_xgb_fast['test_mae']} | Size: {res_xgb_fast['artifact_size_mb']}MB | Throughput: {res_xgb_fast['throughput_samples_per_sec']:.0f} s/s | Train: {res_xgb_fast['training_time_sec']}s")

        # 3. Ridge Reference
        if "Ridge" in baselines:
            ridge_base = baselines["Ridge"]
            ridge_base["model_name"] = "Ridge_Reference (L2 alpha 10.0)"
            horizon_runs.append(ridge_base)
            print(f"  [Ref] {ridge_base['model_name']} | MAE: {ridge_base['test_mae']} | Size: {ridge_base['artifact_size_mb']}MB | Throughput: {ridge_base['throughput_samples_per_sec']:.0f} s/s")

        # 4. PyTorch LSTM: Baseline + Compact (1 layer, hidden 32)
        if "LSTM" in baselines:
            lstm_base = baselines["LSTM"]
            lstm_base["model_name"] = "LSTM_Baseline (2 layers, hidden 64)"
            horizon_runs.append(lstm_base)
            print(f"  [Ref] {lstm_base['model_name']} | MAE: {lstm_base['test_mae']} | Size: {lstm_base['artifact_size_mb']}MB | Throughput: {lstm_base['throughput_samples_per_sec']:.0f} s/s")

        compact_lstm = LSTMForecaster(
            name=f"lstm_compact_h{h}", hidden_dim=32, num_layers=1, dropout=0.1, max_epochs=6, batch_size=1024, random_state=42
        )
        print("  ---> Benchmarking LSTM_Compact (1 layer, hidden 32, batch 1024)...", end="", flush=True)
        res_lstm_compact = profile_model_efficiency(compact_lstm, X_train, y_train, X_val, y_val, X_test, y_test, "LSTM_Compact (1 layer, hidden 32)", temp_path)
        horizon_runs.append(res_lstm_compact)
        print(f" Done! Test MAE: {res_lstm_compact['test_mae']} | Size: {res_lstm_compact['artifact_size_mb']}MB | Throughput: {res_lstm_compact['throughput_samples_per_sec']:.0f} s/s | Train: {res_lstm_compact['training_time_sec']}s")

        complexity_results[f"h{h}"] = horizon_runs

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Save metrics JSON
    out_json = {
        "experiment": "Phase 11 Stage 3: Model Complexity Optimization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": complexity_results,
    }
    out_json_path = Path("artifacts/metrics/model_complexity_optimization.json")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[3/4] Exported model complexity metrics to '{out_json_path}'")

    # Generate Publication Plot
    print(f"\n[4/4] Generating visualization at 'artifacts/plots/model_accuracy_vs_complexity.png'...")
    plot_complexity_tradeoffs(complexity_results, "artifacts/plots/model_accuracy_vs_complexity.png")
    print("Done!")


def plot_complexity_tradeoffs(results: Dict[str, Any], output_path: str):
    """Plot MAE vs Model Size and MAE vs Inference Throughput."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    horizons = [1, 6, 24]

    for idx, h in enumerate(horizons):
        data = results[f"h{h}"]
        df = pd.DataFrame(data)

        # Plot Throughput vs Test MAE
        for _, row in df.iterrows():
            clean_name = row["model_name"].split(" (")[0]
            # Sizing bubble by artifact size (min 40, max 350)
            bsize = max(40, min(350, float(row["artifact_size_mb"]) * 10))
            axes[idx].scatter(
                row["throughput_samples_per_sec"], row["test_mae"],
                s=bsize, alpha=0.8, edgecolors="black", linewidth=1.2,
                label=clean_name
            )
            short_lbl = clean_name.replace("RandomForest", "RF").replace("LinearRegression", "OLS")
            axes[idx].annotate(
                short_lbl,
                (row["throughput_samples_per_sec"], row["test_mae"]),
                fontsize=8, ha="left", va="bottom", xytext=(4, 4), textcoords="offset points"
            )

        axes[idx].set_title(f"Horizon +{h}h: Accuracy vs. Throughput\n(Bubble size = Model Artifact Size)", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Inference Throughput (Samples / Sec)", fontsize=10)
        axes[idx].set_ylabel(r"Test MAE ($\mu g/m^3$)", fontsize=10)
        axes[idx].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
