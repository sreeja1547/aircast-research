"""Comprehensive reconciliation and cross-verification script for Phase 11."""

import json
from pathlib import Path
import os

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

print("="*70)
print("1. COMPARING EXPERIMENT 1 AND EFFICIENCY BASELINE PREDICTIVE METRICS")
print("="*70)
exp1 = load_json("artifacts/metrics/experiment1_benchmarking.json")
eff = load_json("artifacts/metrics/efficiency_baseline.json")

exp1_map = {(item["model_name"], item["horizon_hours"]): item for item in exp1["results"]}
eff_map = {(item["model_name"], item["horizon_hours"]): item for item in eff["results"]}

print(f"{'Model':22s} | {'H':2s} | {'Exp1 MAE':9s} | {'Eff MAE':9s} | {'Exp1 RMSE':9s} | {'Eff RMSE':9s} | {'Match?'}")
print("-" * 80)
for key in sorted(exp1_map.keys()):
    m, h = key
    e1 = exp1_map[key]
    eb = eff_map.get(key)
    e1_mae = e1["mae"]
    e1_rmse = e1["rmse"]
    eb_mae = eb["predictive_performance"]["mae"] if eb else None
    eb_rmse = eb["predictive_performance"]["rmse"] if eb else None
    mae_diff = abs(e1_mae - eb_mae) if eb else 999
    match_str = "EXACT MATCH" if mae_diff < 1e-4 else f"DIFF: {mae_diff:.4f}"
    print(f"{m:22s} | {h:2d} | {e1_mae:9.4f} | {eb_mae:9.4f} | {e1_rmse:9.4f} | {eb_rmse:9.4f} | {match_str}")

print("\n" + "="*70)
print("2. FEATURE EFFICIENCY & PRUNING (STAGE 2)")
print("="*70)
feat_opt = load_json("artifacts/metrics/feature_efficiency_comparison.json")
for h in ["h1", "h6", "h24"]:
    hdata = feat_opt["results"][h]
    b = hdata["baseline_47_features"]
    o = hdata["optimized_compact_features"]
    c = hdata["comparison"]
    print(f"Horizon {h}:")
    print(f"  Baseline:  {b['n_features']} features, Test MAE={b['test_mae']}, RMSE={b['test_rmse']}, Latency={b['mean_per_sample_latency_ms']:.4f} ms, TrainTime={b['training_time_sec']:.2f} s")
    print(f"  Optimized: {o['n_features']} features, Test MAE={o['test_mae']}, RMSE={o['test_rmse']}, Latency={o['mean_per_sample_latency_ms']:.4f} ms, TrainTime={o['training_time_sec']:.2f} s")
    print(f"  Delta MAE: {c['delta_mae']:+.3f}, Speedup: {c['training_speedup_factor']}x, RAM Savings: {c['ram_savings_pct']}%")

print("\n" + "="*70)
print("3. MODEL COMPLEXITY OPTIMIZATION (STAGE 3)")
print("="*70)
comp_opt = load_json("artifacts/metrics/model_complexity_optimization.json")
for h, mlist in comp_opt.get("results", {}).items():
    print(f"\nHorizon {h}:")
    for item in mlist:
        print(f"  {item['model_name']:45s} | MAE={item['test_mae']:.3f} | RMSE={item['test_rmse']:.3f} | Size={item['artifact_size_mb']:.3f} MB | TrainTime={item['training_time_sec']:.2f} s")

print("\n" + "="*70)
print("4. TRAINING OPTIMIZATION (STAGE 4)")
print("="*70)
train_opt = load_json("artifacts/metrics/training_efficiency.json")
print("Precision Optimization:")
print(json.dumps(train_opt.get("precision_optimization"), indent=2))
print("\nDataLoader Tuning:")
print(json.dumps(train_opt.get("dataloader_tuning"), indent=2))
print("\nThreading Scaling:")
print(json.dumps(train_opt.get("threading_scaling"), indent=2))
print("\nVectorization:")
print(json.dumps(train_opt.get("vectorization"), indent=2))

print("\n" + "="*70)
print("5. INFERENCE OPTIMIZATION (STAGE 5)")
print("="*70)
inf_opt = load_json("artifacts/metrics/inference_optimization.json")
print("Single Sample Benchmarks:")
print(json.dumps(inf_opt.get("single_sample_benchmarks"), indent=2))

print("\nBatch Scaling Peak Throughput:")
for mname, blist in inf_opt.get("batch_scaling_benchmarks", {}).items():
    last = blist[-1]
    first = blist[0]
    print(f"  {mname:20s}: bs=1 thr={first.get('throughput_samples_per_sec'):.1f} s/s | peak bs={last.get('batch_size')} thr={last.get('throughput_samples_per_sec'):.1f} s/s")

print("\n" + "="*70)
print("6. MEMORY OPTIMIZATION & COMPRESSION (STAGE 6)")
print("="*70)
mem_opt = load_json("artifacts/metrics/memory_optimization.json")
print("Dataset Downcasting:")
print(json.dumps(mem_opt.get("dataset_downcasting"), indent=2))
print("\nParquet Codecs:")
print(json.dumps(mem_opt.get("parquet_codecs"), indent=2))
print("\nMemory Leak Verification:")
print(json.dumps(mem_opt.get("memory_leak_verification"), indent=2))

print("\n" + "="*70)
print("7. HORIZON-SPECIFIC MODEL SELECTION & PARETO (STAGE 7)")
print("="*70)
horiz_opt = load_json("artifacts/metrics/horizon_specific_model_selection.json")
for h, hdict in horiz_opt.get("results", {}).items():
    print(f"\nHorizon {h} Pareto Frontier:")
    for p in hdict.get("pareto_frontier", []):
        print(f"  {p['model_name']:45s} | MAE={p['test_mae']:.3f} | Size={p['artifact_size_mb']:.3f} MB | Latency={p['mean_latency_ms']:.4f} ms | Score={p.get('composite_utility_score', 0):.4f}")

print("\nSelected Horizon Suite:")
for h, sdata in horiz_opt.get("suite_summary", {}).get("selected_horizon_suite", {}).items():
    print(f"  {h}: {sdata['model_name']} | MAE={sdata['test_mae']} | RMSE={sdata['test_rmse']} | Size={sdata['artifact_size_mb']} MB")

print("\n" + "="*70)
print("8. API AND DASHBOARD PERFORMANCE (STAGE 8)")
print("="*70)
api_perf = load_json("artifacts/metrics/api_performance.json")
dash_perf = load_json("artifacts/metrics/dashboard_performance.json")
print("API Performance Benchmarks:")
for ep, ep_data in api_perf.get("benchmarks", {}).items():
    print(f"  {ep:25s}: mean={ep_data.get('mean_ms', 0):.3f} ms, p95={ep_data.get('p95_ms', 0):.3f} ms, thr={ep_data.get('throughput_req_per_sec', 0):.1f} req/s")

print("\nDashboard Performance Benchmarks:")
for page, p_data in dash_perf.get("benchmarks", {}).items():
    print(f"  {page:25s}: compute={p_data.get('compute_mean_ms', 0):.3f} ms, total={p_data.get('total_render_mean_ms', 0):.3f} ms, memory_delta={p_data.get('memory_delta_mb', 0):.3f} MB")

print("\n" + "="*70)
print("9. REGRESSION TESTS (STAGE 9)")
print("="*70)
reg_tests = load_json("artifacts/metrics/efficiency_regression_tests.json")
print(json.dumps(reg_tests.get("summary"), indent=2))
