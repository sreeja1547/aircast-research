"""Detailed inspection script for Phase 11 Final Scientific and Performance Reconciliation."""

import json
from pathlib import Path
import os

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

print("=== 1. BASELINE MAE COMPARISONS ===")
exp1 = load_json("artifacts/metrics/experiment1_benchmarking.json")
eff = load_json("artifacts/metrics/efficiency_baseline.json")

print("\n--- Experiment 1 Benchmarking ---")
for h_key in ["1h", "6h", "24h"]:
    print(f"Horizon {h_key}:")
    for m, vals in exp1.get(h_key, {}).items():
        mae = vals.get("test_mae", vals.get("mae"))
        rmse = vals.get("test_rmse", vals.get("rmse"))
        r2 = vals.get("test_r2", vals.get("r2"))
        print(f"  {m:25s}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

print("\n--- Efficiency Baseline ---")
for m, h_dict in eff.get("models", {}).items():
    for h, vals in h_dict.items():
        mae = vals.get("test_mae")
        rmse = vals.get("test_rmse")
        r2 = vals.get("test_r2")
        print(f"  {m:25s} h={h}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

print("\n=== 2. INDIVIDUAL MODEL ARTIFACT SIZES ON DISK ===")
models_dir = Path("artifacts/models")
sizes = {}
total_disk_bytes = 0
for f in sorted(models_dir.glob("*.joblib")):
    s = f.stat().st_size
    sizes[f.name] = s
    total_disk_bytes += s
    print(f"  {f.name:35s}: {s / (1024*1024):.4f} MB ({s:,} bytes)")

print(f"\nTotal artifacts on disk: {total_disk_bytes / (1024*1024):.4f} MB ({total_disk_bytes:,} bytes)")

print("\n=== 3. RF SIZES AND COMPRESSION ===")
rf_total_bytes = sum(sizes[f"random_forest_h{h}.joblib"] for h in [1, 6, 24] if f"random_forest_h{h}.joblib" in sizes)
print(f"RF uncompressed baseline total: {rf_total_bytes / (1024*1024):.4f} MB")

mem_opt = load_json("artifacts/metrics/memory_optimization.json")
print("Memory optimization RF compression results:")
print(json.dumps(mem_opt.get("model_compression", {}), indent=2))

model_comp = load_json("artifacts/metrics/model_complexity_optimization.json")
print("\nModel complexity optimization RF results:")
for h, mdict in model_comp.get("results", {}).items():
    print(f"Horizon {h}:")
    for variant, vvals in mdict.get("Random Forest", {}).items():
        print(f"  {variant}: size_mb={vvals.get('size_mb')}, test_mae={vvals.get('test_mae')}, trees={vvals.get('n_estimators')}, max_depth={vvals.get('max_depth')}")

print("\n=== 4. SUITE COMPARISONS & SELECTION INCONSISTENCIES ===")
horiz_sel = load_json("artifacts/metrics/horizon_specific_model_selection.json")
print("Horizon Selection Suite:")
print(json.dumps(horiz_sel.get("suite_summary", {}), indent=2))

fin_sum = load_json("artifacts/metrics/final_efficiency_summary.json")
print("\nFinal Efficiency Summary before_vs_after:")
print(json.dumps(fin_sum.get("before_vs_after_comparison", {}), indent=2))
