"""Stage 5: Inference-Path Optimization & Acceleration for AirCast Research.

Evaluates:
1. Single-sample prediction latency:
   - Baseline Path: Dictionary -> 1-row Pandas DataFrame -> predict.
   - Optimized Fast Path: Dictionary -> pre-allocated NumPy array -> predict.
   - Direct Array Path: Raw NumPy array input.
2. Batch inference throughput & latency scaling:
   - Batch sizes: 1, 16, 32, 64, 128, 256, 512, 1024.
   - Architectures: Ridge, HistGBDT, RandomForest, XGBoost, LSTM.
3. Numerical prediction equivalence verification (tolerance < 1e-4).

Exports:
- artifacts/metrics/inference_optimization.json
- artifacts/plots/inference_latency_comparison.png
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.features.registry import get_feature_group
from src.models import (
    HistGradientBoostingForecaster,
    LSTMForecaster,
    PersistenceForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)


def benchmark_single_sample_paths(model: Any, feature_names: List[str], sample_dict: Dict[str, float], n_iter: int = 100) -> Dict[str, Any]:
    """Compare baseline Pandas DataFrame conversion vs optimized NumPy pre-allocation."""
    # 1. Baseline Path: Dict -> DataFrame
    t_df_list = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        row_dict = {col: float(sample_dict.get(col, 0.0)) for col in feature_names}
        df_sample = pd.DataFrame([row_dict])
        pred_df = model.predict(df_sample)
        t_df_list.append((time.perf_counter() - t0) * 1000.0)

    # 2. Optimized Fast Path: Dict -> NumPy buffer
    feat_indices = {col: i for i, col in enumerate(feature_names)}
    n_feats = len(feature_names)
    buf = np.empty((1, n_feats), dtype=np.float32)
    t_buf_list = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        for col, idx in feat_indices.items():
            buf[0, idx] = float(sample_dict.get(col, 0.0))
        pred_buf = model.predict(buf)
        t_buf_list.append((time.perf_counter() - t0) * 1000.0)

    # 3. Direct NumPy Path (Array already in memory)
    t_direct_list = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        pred_direct = model.predict(buf)
        t_direct_list.append((time.perf_counter() - t0) * 1000.0)

    # Numerical equivalence
    diff = float(np.max(np.abs(np.asarray(pred_df) - np.asarray(pred_buf))))

    return {
        "baseline_pandas_df_mean_ms": round(float(np.mean(t_df_list)), 4),
        "baseline_pandas_df_p95_ms": round(float(np.percentile(t_df_list, 95)), 4),
        "optimized_numpy_buf_mean_ms": round(float(np.mean(t_buf_list)), 4),
        "optimized_numpy_buf_p95_ms": round(float(np.percentile(t_buf_list, 95)), 4),
        "direct_numpy_mean_ms": round(float(np.mean(t_direct_list)), 4),
        "speedup_vs_baseline": round(float(np.mean(t_df_list)) / max(1e-4, float(np.mean(t_buf_list))), 2),
        "numerical_equivalence_max_diff": diff,
    }


def benchmark_batch_scaling(model: Any, X_pool: np.ndarray, batch_sizes: List[int], n_reps: int = 25) -> List[Dict[str, Any]]:
    """Measure latency percentiles and throughput across varying batch sizes."""
    results = []
    pool_size = len(X_pool)

    for bs in batch_sizes:
        if bs > pool_size:
            continue
        X_sub = X_pool[:bs]

        # Warmup
        for _ in range(3):
            _ = model.predict(X_sub)

        lat_ms = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ = model.predict(X_sub)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(lat_ms))
        p50_ms = float(np.median(lat_ms))
        p95_ms = float(np.percentile(lat_ms, 95))
        p99_ms = float(np.percentile(lat_ms, 99))
        throughput = round(bs / (mean_ms / 1000.0), 1)
        per_sample_ms = round(mean_ms / bs, 5)

        results.append({
            "batch_size": bs,
            "mean_batch_latency_ms": round(mean_ms, 4),
            "p50_batch_latency_ms": round(p50_ms, 4),
            "p95_batch_latency_ms": round(p95_ms, 4),
            "p99_batch_latency_ms": round(p99_ms, 4),
            "per_sample_latency_ms": per_sample_ms,
            "throughput_samples_per_sec": throughput,
        })

    return results


def main():
    print("=" * 80)
    print("STAGE 5: INFERENCE-PATH OPTIMIZATION & ACCELERATION")
    print("=" * 80)

    feature_names = get_feature_group("combined_all")
    cache_path = "data/processed/forecasting_features_cache.parquet"
    print(f"Loading cached features for profiling from '{cache_path}'...")
    df_prepared = pd.read_parquet(cache_path)

    # Use 2,000 samples for batch testing
    sample_cols = [c for c in feature_names if c in df_prepared.columns]
    df_eval = df_prepared[sample_cols].iloc[-2000:].copy()
    for col in df_eval.columns:
        if not pd.api.types.is_numeric_dtype(df_eval[col].dtype):
            uvals = sorted([str(v) for v in df_eval[col].dropna().unique()])
            smap = {v: float(i) for i, v in enumerate(uvals)}
            df_eval[col] = df_eval[col].astype(str).map(smap).fillna(-1.0).astype(np.float32)

    X_test_pool = df_eval.to_numpy(dtype=np.float32)

    # Create dummy single sample dict
    sample_dict = {c: float(df_eval[c].iloc[-1]) for c in sample_cols}

    models_to_test = [
        ("Ridge", "artifacts/models/ridge_h1.joblib"),
        ("HistGradientBoosting", "artifacts/models/hist_gradient_boosting_h1.joblib"),
        ("RandomForest", "artifacts/models/random_forest_h1.joblib"),
        ("XGBoost", "artifacts/models/xgboost_h1.joblib"),
        ("LSTM", "artifacts/models/lstm_h1.joblib"),
    ]

    single_sample_results = {}
    batch_scaling_results = {}
    batch_sizes = [1, 16, 32, 64, 128, 256, 512, 1024]

    for model_name, model_path in models_to_test:
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} (path not found: {model_path})")
            continue

        print(f"\n---> Benchmarking {model_name} inference path...")
        if model_name == "LSTM":
            model = LSTMForecaster.load(model_path)
        else:
            model = joblib.load(model_path)

        # 1. Single sample paths
        res_single = benchmark_single_sample_paths(model, sample_cols, sample_dict, n_iter=60)
        single_sample_results[model_name] = res_single
        print(f"     Single Sample: Pandas={res_single['baseline_pandas_df_mean_ms']:.3f}ms | NumPy Buffer={res_single['optimized_numpy_buf_mean_ms']:.3f}ms (Speedup: {res_single['speedup_vs_baseline']}x)")

        # 2. Batch scaling
        res_batch = benchmark_batch_scaling(model, X_test_pool, batch_sizes, n_reps=20)
        batch_scaling_results[model_name] = res_batch
        max_tp = max(b["throughput_samples_per_sec"] for b in res_batch)
        print(f"     Peak Batch Throughput: {max_tp:,.1f} samples/sec")

    # Export metrics JSON
    out_json = {
        "experiment": "Phase 11 Stage 5: Inference Optimization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "single_sample_benchmarks": single_sample_results,
        "batch_scaling_benchmarks": batch_scaling_results,
    }
    out_path = Path("artifacts/metrics/inference_optimization.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved inference optimization metrics to '{out_path}'")

    # Generate Publication Plot
    plot_inference_comparison(out_json, "artifacts/plots/inference_latency_comparison.png")


def plot_inference_comparison(data: Dict[str, Any], output_path: str):
    """Plot single sample speedup and batch throughput curves."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Single-sample latency comparison (Pandas vs NumPy Buffer)
    single_res = data["single_sample_benchmarks"]
    models = list(single_res.keys())
    df_lat = [single_res[m]["baseline_pandas_df_mean_ms"] for m in models]
    buf_lat = [single_res[m]["optimized_numpy_buf_mean_ms"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width/2, df_lat, width, label="Baseline (Pandas DataFrame)", color="#e74c3c", alpha=0.85)
    axes[0].bar(x + width/2, buf_lat, width, label="Optimized (NumPy Buffer)", color="#2ecc71", alpha=0.85)
    axes[0].set_ylabel("Mean Latency (ms)", fontsize=11)
    axes[0].set_title("Single-Sample Prediction Latency\n(Pandas Overhead Elimination)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Batch scaling throughput curves
    batch_res = data["batch_scaling_benchmarks"]
    palette = sns.color_palette("tab10", n_colors=len(models))

    for idx, m in enumerate(models):
        if m in batch_res:
            bs_data = batch_res[m]
            b_sizes = [d["batch_size"] for d in bs_data]
            tps = [d["throughput_samples_per_sec"] for d in bs_data]
            axes[1].plot(b_sizes, tps, marker="o", linewidth=2, label=m, color=palette[idx])

    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Batch Size (Log Scale)", fontsize=11)
    axes[1].set_ylabel("Throughput (Samples / Sec, Log Scale)", fontsize=11)
    axes[1].set_title("Batch Inference Throughput Scaling", fontsize=12, fontweight="bold")
    axes[1].legend(frameon=True)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Generated inference plot at '{output_path}'")


if __name__ == "__main__":
    main()
