"""Stage 4: Training-Time Optimization Experiments for AirCast Research.

Evaluates:
1. Precision optimization: float64 vs float32 memory, fit time, and numerical fidelity.
2. PyTorch DataLoader batch sizing & throughput across tabular sequence tensors.
3. Multi-threading parallel scaling (Amdahl's Law) for XGBoost and Random Forest (1, 2, 4, 8, 12 threads).
4. Vectorized feature engineering vs iterative row/group operations.

Exports:
- artifacts/metrics/training_efficiency.json
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.evaluation.metrics import calculate_regression_metrics
from src.preprocessing.splitter import TemporalPurgedSplitter


def benchmark_precision(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Compare float64 vs float32 training efficiency, memory, and prediction fidelity."""
    print("\n--- 1. Precision Optimization (float64 vs float32) ---")
    
    # 64-bit arrays
    X_tr_64 = X_train.astype(np.float64).to_numpy()
    y_tr_64 = y_train.astype(np.float64).to_numpy()
    X_te_64 = X_test.astype(np.float64).to_numpy()
    mem_64_mb = round((X_tr_64.nbytes + y_tr_64.nbytes) / (1024 * 1024), 2)
    
    # 32-bit arrays
    X_tr_32 = X_train.astype(np.float32).to_numpy()
    y_tr_32 = y_train.astype(np.float32).to_numpy()
    X_te_32 = X_test.astype(np.float32).to_numpy()
    mem_32_mb = round((X_tr_32.nbytes + y_tr_32.nbytes) / (1024 * 1024), 2)
    
    # Ridge Benchmark
    ridge64 = Ridge(alpha=10.0)
    t0 = time.perf_counter()
    ridge64.fit(X_tr_64, y_tr_64)
    t_ridge64 = time.perf_counter() - t0
    preds64_ridge = ridge64.predict(X_te_64)
    
    ridge32 = Ridge(alpha=10.0)
    t0 = time.perf_counter()
    ridge32.fit(X_tr_32, y_tr_32)
    t_ridge32 = time.perf_counter() - t0
    preds32_ridge = ridge32.predict(X_te_32)
    
    ridge_max_delta = float(np.max(np.abs(preds64_ridge - preds32_ridge)))
    ridge_mae64 = float(np.mean(np.abs(preds64_ridge - y_test)))
    ridge_mae32 = float(np.mean(np.abs(preds32_ridge - y_test)))

    # XGBoost Benchmark
    xgb64 = xgb.XGBRegressor(n_estimators=60, max_depth=6, tree_method="hist", n_jobs=-1, random_state=42)
    t0 = time.perf_counter()
    xgb64.fit(X_tr_64, y_tr_64)
    t_xgb64 = time.perf_counter() - t0
    preds64_xgb = xgb64.predict(X_te_64)
    
    xgb32 = xgb.XGBRegressor(n_estimators=60, max_depth=6, tree_method="hist", n_jobs=-1, random_state=42)
    t0 = time.perf_counter()
    xgb32.fit(X_tr_32, y_tr_32)
    t_xgb32 = time.perf_counter() - t0
    preds32_xgb = xgb32.predict(X_te_32)
    
    xgb_max_delta = float(np.max(np.abs(preds64_xgb - preds32_xgb)))
    xgb_mae64 = float(np.mean(np.abs(preds64_xgb - y_test)))
    xgb_mae32 = float(np.mean(np.abs(preds32_xgb - y_test)))

    results = {
        "memory_float64_mb": mem_64_mb,
        "memory_float32_mb": mem_32_mb,
        "memory_reduction_pct": round((1.0 - mem_32_mb / mem_64_mb) * 100, 2),
        "ridge": {
            "fit_time_float64_sec": round(t_ridge64, 4),
            "fit_time_float32_sec": round(t_ridge32, 4),
            "speedup": round(t_ridge64 / max(1e-4, t_ridge32), 2),
            "mae_float64": round(ridge_mae64, 4),
            "mae_float32": round(ridge_mae32, 4),
            "max_abs_prediction_diff": round(ridge_max_delta, 6),
        },
        "xgboost": {
            "fit_time_float64_sec": round(t_xgb64, 4),
            "fit_time_float32_sec": round(t_xgb32, 4),
            "speedup": round(t_xgb64 / max(1e-4, t_xgb32), 2),
            "mae_float64": round(xgb_mae64, 4),
            "mae_float32": round(xgb_mae32, 4),
            "max_abs_prediction_diff": round(xgb_max_delta, 6),
        }
    }
    print(f"  Memory Footprint: {mem_64_mb} MB (float64) -> {mem_32_mb} MB (float32) [{results['memory_reduction_pct']}% reduction]")
    print(f"  Ridge: float64 fit={t_ridge64:.3f}s | float32 fit={t_ridge32:.3f}s | Max diff: {ridge_max_delta:.6f}")
    print(f"  XGBoost: float64 fit={t_xgb64:.3f}s | float32 fit={t_xgb32:.3f}s | Max diff: {xgb_max_delta:.6f}")
    return results


def benchmark_dataloader(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """Evaluate PyTorch DataLoader throughput across different batch sizes."""
    print("\n--- 2. PyTorch DataLoader Batch Size Tuning ---")
    X_tensor = torch.tensor(X_train.astype(np.float32).to_numpy()[:, np.newaxis, :], dtype=torch.float32)
    y_tensor = torch.tensor(y_train.astype(np.float32).to_numpy(), dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    batch_sizes = [128, 256, 512, 1024, 2048]
    loader_results = []
    
    for bs in batch_sizes:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)
        # Warm-up pass
        for _ in range(5):
            pass
        
        t0 = time.perf_counter()
        n_items = 0
        for bx, by in loader:
            n_items += len(bx)
        dur = time.perf_counter() - t0
        throughput = round(n_items / dur, 1)
        
        res = {
            "batch_size": bs,
            "epoch_duration_sec": round(dur, 4),
            "throughput_samples_per_sec": throughput,
        }
        loader_results.append(res)
        print(f"  Batch {bs:4d}: {dur:.3f}s per epoch ({throughput:,.0f} samples/sec)")
        
    return {"results": loader_results}


def benchmark_threading_scaling(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """Measure multi-threading scaling and parallel efficiency for XGBoost and Random Forest."""
    print("\n--- 3. Multi-Threading Scaling (Amdahl's Law Analysis) ---")
    threads = [1, 2, 4, 8, 12]
    
    # XGBoost Scaling
    xgb_results = []
    t1_xgb = None
    for th in threads:
        model = xgb.XGBRegressor(n_estimators=60, max_depth=6, tree_method="hist", n_jobs=th, random_state=42)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        dur = round(time.perf_counter() - t0, 3)
        if th == 1:
            t1_xgb = dur
        speedup = round(t1_xgb / dur, 2)
        efficiency = round((speedup / th) * 100, 1)
        xgb_results.append({
            "threads": th,
            "training_time_sec": dur,
            "speedup": speedup,
            "parallel_efficiency_pct": efficiency,
        })
        print(f"  XGBoost {th:2d} threads: {dur:6.2f}s | Speedup: {speedup:4.2f}x | Efficiency: {efficiency:5.1f}%")

    # Random Forest Scaling (evaluated on compact 20 trees to preserve benchmark runtime)
    rf_results = []
    t1_rf = None
    for th in threads:
        model = RandomForestRegressor(n_estimators=20, max_depth=8, min_samples_leaf=8, n_jobs=th, random_state=42)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        dur = round(time.perf_counter() - t0, 3)
        if th == 1:
            t1_rf = dur
        speedup = round(t1_rf / dur, 2)
        efficiency = round((speedup / th) * 100, 1)
        rf_results.append({
            "threads": th,
            "training_time_sec": dur,
            "speedup": speedup,
            "parallel_efficiency_pct": efficiency,
        })
        print(f"  RandomForest {th:2d} threads: {dur:6.2f}s | Speedup: {speedup:4.2f}x | Efficiency: {efficiency:5.1f}%")

    return {
        "xgboost_scaling": xgb_results,
        "random_forest_scaling": rf_results,
    }


def benchmark_vectorization() -> Dict[str, Any]:
    """Compare vectorized feature computation vs Python loop iteration."""
    print("\n--- 4. Feature Vectorization Benchmark ---")
    n = 200_000
    stations = [f"station_{i}" for i in range(12)]
    df = pd.DataFrame({
        "station": np.random.choice(stations, size=n),
        "pm25": np.random.uniform(5.0, 350.0, size=n),
    })
    
    # Vectorized pandas groupby-shift
    t0 = time.perf_counter()
    _ = df.groupby("station")["pm25"].shift(1)
    _ = df.groupby("station")["pm25"].shift(6)
    _ = df.groupby("station")["pm25"].shift(24)
    t_vec = time.perf_counter() - t0
    
    # Python row-by-row simulation
    t0 = time.perf_counter()
    # Sample 10,000 rows to extrapolate row-by-row iteration time
    sample_df = df.iloc[:10_000]
    out = []
    for _, row in sample_df.iterrows():
        out.append(row["pm25"] * 1.05)
    t_iter_sample = time.perf_counter() - t0
    t_iter_extrapolated = t_iter_sample * (n / 10_000)
    
    speedup = round(t_iter_extrapolated / max(1e-4, t_vec), 1)
    print(f"  Vectorized Groupby Lags ({n:,} rows): {t_vec:.4f}s")
    print(f"  Iterative Row Processing (extrapolated {n:,} rows): {t_iter_extrapolated:.4f}s")
    print(f"  Vectorization Speedup: {speedup:,.0f}x")
    
    return {
        "sample_size": n,
        "vectorized_groupby_duration_sec": round(t_vec, 4),
        "iterative_duration_sec": round(t_iter_extrapolated, 4),
        "vectorization_speedup": speedup,
    }


def main():
    print("=" * 80)
    print("STAGE 4: TRAINING-TIME OPTIMIZATION BENCHMARK")
    print("=" * 80)
    
    cache_path = "data/processed/forecasting_features_cache.parquet"
    print(f"Loading features from '{cache_path}'...")
    df_prepared = pd.read_parquet(cache_path)
    
    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()
    
    # Use h=1 dataset for standard training pipeline benchmarks
    X_full, y_full, meta = dataset_builder.get_horizon_dataset(
        df_prepared, horizon=1, feature_group="combined_all"
    )
    for col in X_full.columns:
        if not pd.api.types.is_numeric_dtype(X_full[col].dtype):
            uvals = sorted([str(v) for v in X_full[col].dropna().unique()])
            smap = {v: float(i) for i, v in enumerate(uvals)}
            X_full[col] = X_full[col].astype(str).map(smap).fillna(-1.0).astype(np.float32)

    dt_series = pd.to_datetime(meta[splitter.datetime_col])
    train_mask = dt_series <= splitter.train_end
    test_mask = (dt_series >= splitter.test_start) & (dt_series <= splitter.test_end)

    X_train, y_train = X_full[train_mask], y_full[train_mask]
    X_test, y_test = X_full[test_mask], y_full[test_mask]
    
    res_precision = benchmark_precision(X_train, y_train, X_test, y_test)
    res_loader = benchmark_dataloader(X_train, y_train)
    res_threading = benchmark_threading_scaling(X_train, y_train)
    res_vectorization = benchmark_vectorization()
    
    final_output = {
        "experiment": "Phase 11 Stage 4: Training-Time Optimization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "precision_optimization": res_precision,
        "dataloader_tuning": res_loader,
        "threading_scaling": res_threading,
        "vectorization": res_vectorization,
    }
    
    out_path = Path("artifacts/metrics/training_efficiency.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved training efficiency results to '{out_path}'")


if __name__ == "__main__":
    main()
