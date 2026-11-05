"""Stage 1: Establish Performance Baselines for AirCast Research.

Measures:
- Predictive performance (MAE, RMSE, R2)
- Training & validation duration, parameters / boosting rounds
- Inference latency: single-sample, batch-32, batch-128, batch-512 (mean, median, P95, std)
- Throughput (predictions/sec)
- Peak process RAM, GPU memory (if applicable), model file sizes
- Feature generation time vs model prediction time
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
import psutil

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.evaluation.metrics import calculate_regression_metrics
from src.models.deep_learning import LSTMForecaster
from src.preprocessing.splitter import TemporalPurgedSplitter


def profile_batch_latencies(model: Any, X_data: pd.DataFrame, batch_sizes: List[int], n_warmup: int = 5, n_runs: int = 50) -> Dict[str, Any]:
    """Measure single-sample and batch latencies with warmup and repeated trials."""
    results = {}

    # 1. Single sample latency
    single_sample = X_data.iloc[[0]]
    for _ in range(n_warmup):
        _ = model.predict(single_sample)

    single_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model.predict(single_sample)
        t1 = time.perf_counter()
        single_times.append((t1 - t0) * 1000.0)  # ms

    single_arr = np.array(single_times)
    results["single_sample"] = {
        "mean_ms": round(float(np.mean(single_arr)), 4),
        "median_ms": round(float(np.median(single_arr)), 4),
        "p95_ms": round(float(np.percentile(single_arr, 95)), 4),
        "std_ms": round(float(np.std(single_arr)), 4),
        "min_ms": round(float(np.min(single_arr)), 4),
        "max_ms": round(float(np.max(single_arr)), 4),
    }

    # 2. Batch latencies
    for b in batch_sizes:
        if len(X_data) < b:
            continue
        batch_slice = X_data.iloc[:b]

        for _ in range(n_warmup):
            _ = model.predict(batch_slice)

        batch_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model.predict(batch_slice)
            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000.0)  # ms

        b_arr = np.array(batch_times)
        throughput = float(b / (np.mean(b_arr) / 1000.0))
        results[f"batch_{b}"] = {
            "batch_size": b,
            "mean_ms": round(float(np.mean(b_arr)), 4),
            "median_ms": round(float(np.median(b_arr)), 4),
            "p95_ms": round(float(np.percentile(b_arr, 95)), 4),
            "std_ms": round(float(np.std(b_arr)), 4),
            "per_sample_ms": round(float(np.mean(b_arr) / b), 5),
            "throughput_samples_per_sec": round(throughput, 1),
        }

    return results


def main():
    print("=" * 80)
    print("STAGE 1: ESTABLISHING RIGOROUS PERFORMANCE BASELINES")
    print("=" * 80)

    process = psutil.Process(os.getpid())
    init_ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[Initial Process RAM]: {init_ram_mb:.2f} MB")

    cache_path = "data/processed/forecasting_features_cache.parquet"
    print(f"\n[1/3] Loading parquet feature cache from '{cache_path}'...")
    t0_load = time.perf_counter()
    df_prepared = pd.read_parquet(cache_path)
    t_load = time.perf_counter() - t0_load
    print(f"      Loaded {len(df_prepared):,} rows in {t_load:.2f}s")

    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    # Preprocessing & feature generation benchmark
    print("\n[2/3] Benchmarking feature-generation and preprocessing slice latency...")
    t0_feat = time.perf_counter()
    X_sample_slice, y_sample_slice, _ = dataset_builder.get_horizon_dataset(
        df_prepared.iloc[:5000], horizon=1, feature_group="combined_all"
    )
    t_feat_ms = ((time.perf_counter() - t0_feat) / 5000) * 1000.0
    print(f"      Feature slicing latency: {t_feat_ms:.4f} ms per sample")

    # Load existing experiment1 metrics for training time reference
    exp1_path = "artifacts/metrics/experiment1_benchmarking.json"
    exp1_data = {}
    if os.path.exists(exp1_path):
        with open(exp1_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            for r in d.get("results", []):
                key = f"{r['model_name'].lower().strip()}_h{r['horizon_hours']}"
                exp1_data[key] = r

    horizons = [1, 6, 24]
    model_keys = [
        ("Persistence", "persistence"),
        ("Linear Regression", "linear_regression"),
        ("Ridge", "ridge"),
        ("HistGradientBoosting", "hist_gradient_boosting"),
        ("Random Forest", "random_forest"),
        ("XGBoost", "xgboost"),
        ("LSTM", "lstm"),
    ]

    all_baseline_results = []
    models_dir = Path("artifacts/models")

    print("\n[3/3] Profiling all 7 architectures across 3 horizons (21 configurations)...")

    for h in horizons:
        print(f"\n" + "-" * 70)
        print(f"--- HORIZON h = {h} HOUR(S) ---")
        print("-" * 70)

        # Get test partition
        X_all, y_all, meta = dataset_builder.get_horizon_dataset(
            df_prepared, horizon=h, feature_group="combined_all"
        )
        dt_series = pd.to_datetime(meta[splitter.datetime_col])
        test_mask = (dt_series >= splitter.test_start) & (dt_series <= splitter.test_end)
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        print(f"Test samples for h={h}: {len(X_test):,}")

        for model_display, model_id in model_keys:
            artifact_file = models_dir / f"{model_id}_h{h}.joblib"
            if not artifact_file.exists():
                print(f"[Warning] Artifact not found: {artifact_file}, skipping.")
                continue

            file_size_mb = artifact_file.stat().st_size / (1024 * 1024)

            # Load model
            t_load_start = time.perf_counter()
            if model_id == "lstm":
                model = LSTMForecaster.load(artifact_file)
            else:
                model = joblib.load(artifact_file)
            load_time_sec = round(time.perf_counter() - t_load_start, 4)

            # Full test prediction
            t_pred_start = time.perf_counter()
            y_pred = model.predict(X_test)
            full_test_pred_sec = round(time.perf_counter() - t_pred_start, 4)

            # Metrics
            reg_metrics = calculate_regression_metrics(y_test, y_pred)

            # Latency profiling
            lat_profiles = profile_batch_latencies(
                model=model,
                X_data=X_test,
                batch_sizes=[32, 128, 512],
                n_warmup=5,
                n_runs=40,
            )

            # Extract model complexity info
            exp_key = f"{model_display.lower().strip()}_h{h}"
            exp_ref = exp1_data.get(exp_key, {})
            training_time_sec = exp_ref.get("training_time_sec", getattr(model, "metadata", {}).get("training_time_sec", 0.0))

            complexity_info = {}
            if hasattr(model, "model"):
                inner = model.model
                if hasattr(inner, "n_estimators") and inner.n_estimators is not None:
                    complexity_info["n_estimators"] = int(inner.n_estimators)
                if hasattr(inner, "max_depth") and inner.max_depth is not None:
                    complexity_info["max_depth"] = int(inner.max_depth)
                if hasattr(inner, "n_iter_") and inner.n_iter_ is not None:
                    try:
                        complexity_info["n_iter"] = int(inner.n_iter_)
                    except Exception:
                        pass
            if hasattr(model, "num_parameters") and model.num_parameters is not None:
                complexity_info["parameters"] = int(model.num_parameters)

            current_ram_mb = process.memory_info().rss / (1024 * 1024)

            entry = {
                "horizon_hours": h,
                "model_name": model_display,
                "model_id": model_id,
                "model_type": model.model_type,
                "artifact_path": str(artifact_file),
                "artifact_size_mb": round(file_size_mb, 4),
                "model_load_time_sec": load_time_sec,
                "training_time_sec": round(float(training_time_sec), 3),
                "complexity": complexity_info,
                "predictive_performance": {
                    "mae": round(reg_metrics["mae"], 4),
                    "rmse": round(reg_metrics["rmse"], 4),
                    "r2": round(reg_metrics["r2"], 4),
                    "smape": round(reg_metrics["smape"], 4),
                    "extreme_mae": round(reg_metrics["extreme_mae"], 4) if reg_metrics["extreme_mae"] is not None else None,
                },
                "latency_benchmarks": lat_profiles,
                "throughput_samples_per_sec": lat_profiles.get("batch_512", {}).get("throughput_samples_per_sec", 0.0),
                "full_test_duration_sec": full_test_pred_sec,
                "test_samples": len(y_test),
                "process_ram_mb": round(current_ram_mb, 2),
                "gpu_memory_mb": 0.0,
            }
            all_baseline_results.append(entry)

            single_p50 = lat_profiles["single_sample"]["median_ms"]
            single_p95 = lat_profiles["single_sample"]["p95_ms"]
            b512_tp = lat_profiles.get("batch_512", {}).get("throughput_samples_per_sec", 0.0)
            print(f"  {model_display:<20} | MAE: {reg_metrics['mae']:>6.2f} | 1-samp p50: {single_p50:>6.3f}ms | p95: {single_p95:>6.3f}ms | TP: {b512_tp:>8.1f} s/s | Size: {file_size_mb:>5.2f}MB")

    peak_ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\n[Final Peak Process RAM]: {peak_ram_mb:.2f} MB")

    output_json = {
        "experiment": "Phase 11 Stage 1: Empirical Performance Baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_group": "combined_all",
        "n_features": 47,
        "feature_slicing_per_sample_ms": round(t_feat_ms, 5),
        "peak_ram_mb": round(peak_ram_mb, 2),
        "gpu_memory_used_mb": 0.0,
        "cuda_available": False,
        "results": all_baseline_results,
    }

    out_path = Path("artifacts/metrics/efficiency_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSuccessfully exported baseline metrics to '{out_path}'")


if __name__ == "__main__":
    main()
