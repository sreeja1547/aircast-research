"""Stage 6: Memory and Storage Optimization for AirCast Research.

Evaluates:
1. Model artifact serialization and compression levels (joblib compress=0, 3, 6).
2. Data memory optimization: 64-bit to 32-bit downcasting across the full Beijing dataset.
3. Parquet storage compression codecs (snappy vs gzip vs zstd vs uncompressed).
4. Process RAM profiling and leak verification across repeated inference cycles.

Exports:
- artifacts/metrics/memory_optimization.json
"""

import gc
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

from src.models import (
    HistGradientBoostingForecaster,
    LSTMForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)


def benchmark_model_compression(model_paths: Dict[str, str], temp_dir: str = "artifacts/models/comp_test") -> Dict[str, Any]:
    """Evaluate serialization compression levels, save/load durations, and compression ratios."""
    print("\n--- 1. Model Artifact Compression Benchmark ---")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    compression_levels = [0, 3, 6]

    for model_name, path_str in model_paths.items():
        if not os.path.exists(path_str):
            continue

        print(f"  Benchmarking compression for: {model_name}...")
        raw_obj = joblib.load(path_str)

        model_results = []
        base_size = None

        for c_lvl in compression_levels:
            out_file = os.path.join(temp_dir, f"{model_name}_c{c_lvl}.joblib")
            
            # Save time
            t0 = time.perf_counter()
            joblib.dump(raw_obj, out_file, compress=c_lvl)
            save_sec = round(time.perf_counter() - t0, 4)
            size_mb = round(os.path.getsize(out_file) / (1024 * 1024), 3)

            # Load time
            t0 = time.perf_counter()
            _ = joblib.load(out_file)
            load_sec = round(time.perf_counter() - t0, 4)

            if c_lvl == 0:
                base_size = size_mb

            comp_ratio = round(base_size / max(1e-4, size_mb), 2)

            model_results.append({
                "compress_level": c_lvl,
                "size_mb": size_mb,
                "save_time_sec": save_sec,
                "load_time_sec": load_sec,
                "compression_ratio": comp_ratio,
            })
            print(f"    compress={c_lvl}: Size={size_mb:.2f} MB ({comp_ratio}x reduction) | Save={save_sec:.3f}s | Load={load_sec:.3f}s")

            # Clean up test artifact
            if os.path.exists(out_file):
                os.remove(out_file)

        results[model_name] = model_results

    return results


def benchmark_dataset_downcasting(cache_path: str = "data/processed/forecasting_features_cache.parquet") -> Dict[str, Any]:
    """Measure RAM footprint before and after 32-bit numeric downcasting."""
    print("\n--- 2. Dataset Memory Downcasting Benchmark ---")
    df = pd.read_parquet(cache_path)
    
    # 64-bit memory
    df64 = df.copy()
    for col in df64.select_dtypes(include=["float32", "float64"]).columns:
        df64[col] = df64[col].astype(np.float64)
    mem_64_mb = round(df64.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    # 32-bit downcast
    df32 = df.copy()
    for col in df32.select_dtypes(include=["float64"]).columns:
        df32[col] = df32[col].astype(np.float32)
    for col in df32.select_dtypes(include=["int64"]).columns:
        df32[col] = df32[col].astype(np.int32)
    mem_32_mb = round(df32.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    savings_pct = round((1.0 - mem_32_mb / mem_64_mb) * 100, 2)
    print(f"  Raw 64-bit Memory: {mem_64_mb:.1f} MB")
    print(f"  Downcasted 32-bit Memory: {mem_32_mb:.1f} MB ({savings_pct}% RAM reduction)")

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "memory_float64_mb": mem_64_mb,
        "memory_float32_mb": mem_32_mb,
        "ram_savings_pct": savings_pct,
    }


def benchmark_parquet_codecs(cache_path: str = "data/processed/forecasting_features_cache.parquet", temp_dir: str = "artifacts/models/comp_test") -> Dict[str, Any]:
    """Evaluate parquet compression codecs (uncompressed, snappy, gzip, zstd)."""
    print("\n--- 3. Parquet Storage Codec Benchmark ---")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cache_path)
    # Downcast float64 to float32
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)

    codecs = ["none", "snappy", "gzip", "zstd"]
    results = []
    base_size = None

    for codec in codecs:
        target = os.path.join(temp_dir, f"test_data_{codec}.parquet")
        comp = None if codec == "none" else codec

        t0 = time.perf_counter()
        df.to_parquet(target, compression=comp, index=False)
        save_sec = round(time.perf_counter() - t0, 4)
        size_mb = round(os.path.getsize(target) / (1024 * 1024), 2)

        t0 = time.perf_counter()
        _ = pd.read_parquet(target)
        load_sec = round(time.perf_counter() - t0, 4)

        if codec == "none":
            base_size = size_mb

        ratio = round(base_size / max(1e-4, size_mb), 2)
        results.append({
            "codec": codec,
            "file_size_mb": size_mb,
            "write_time_sec": save_sec,
            "read_time_sec": load_sec,
            "compression_ratio": ratio,
        })
        print(f"  Codec '{codec}': Size={size_mb:.2f} MB ({ratio}x) | Write={save_sec:.3f}s | Read={load_sec:.3f}s")

        if os.path.exists(target):
            os.remove(target)

    return {"results": results}


def benchmark_leak_stability(model_path: str = "artifacts/models/xgboost_h1.joblib", n_cycles: int = 50) -> Dict[str, Any]:
    """Verify memory stability across repeated batch inference iterations without memory leaks."""
    print("\n--- 4. Memory Leak Verification (Inference Loops) ---")
    process = psutil.Process()
    gc.collect()
    start_ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)

    model = joblib.load(model_path)
    dummy_input = np.random.randn(500, 47).astype(np.float32)

    ram_samples = []
    for cycle in range(n_cycles):
        _ = model.predict(dummy_input)
        if (cycle + 1) % 10 == 0:
            current_ram = round(process.memory_info().rss / (1024 * 1024), 2)
            ram_samples.append(current_ram)

    gc.collect()
    end_ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    net_growth_mb = round(end_ram_mb - start_ram_mb, 2)
    print(f"  Start RAM: {start_ram_mb:.1f} MB | End RAM: {end_ram_mb:.1f} MB | Net Delta: {net_growth_mb:+.2f} MB")

    return {
        "start_ram_mb": start_ram_mb,
        "end_ram_mb": end_ram_mb,
        "net_growth_mb": net_growth_mb,
        "is_memory_stable": abs(net_growth_mb) < 20.0,
    }


def main():
    print("=" * 80)
    print("STAGE 6: MEMORY AND STORAGE OPTIMIZATION")
    print("=" * 80)

    model_paths = {
        "RandomForest": "artifacts/models/random_forest_h1.joblib",
        "XGBoost": "artifacts/models/xgboost_h1.joblib",
        "HistGradientBoosting": "artifacts/models/hist_gradient_boosting_h1.joblib",
        "LSTM": "artifacts/models/lstm_h1.joblib",
    }

    res_comp = benchmark_model_compression(model_paths)
    res_downcast = benchmark_dataset_downcasting()
    res_parquet = benchmark_parquet_codecs()
    res_leak = benchmark_leak_stability()

    final_results = {
        "experiment": "Phase 11 Stage 6: Memory and Storage Optimization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_compression": res_comp,
        "dataset_downcasting": res_downcast,
        "parquet_codecs": res_parquet,
        "memory_leak_verification": res_leak,
    }

    out_path = Path("artifacts/metrics/memory_optimization.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved memory optimization results to '{out_path}'")


if __name__ == "__main__":
    main()
