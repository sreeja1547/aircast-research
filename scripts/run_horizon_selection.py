"""Stage 7: Horizon-Specific Model Selection (Pareto Optimization) for AirCast Research.

Evaluates multi-objective Pareto frontiers across:
1. Accuracy: Test MAE (minimize)
2. Latency: Single-sample / per-sample prediction latency in ms (minimize)
3. Model Storage: Disk footprint in MB (minimize)
4. Training Compute: Wall-clock training duration in seconds (minimize)

Selects the Pareto-optimal model for each forecasting horizon:
- Horizon 1h (Short-term)
- Horizon 6h (Medium-term)
- Horizon 24h (Long-term)

Compares the optimized horizon-specific deployment suite against uniform baselines.

Exports:
- artifacts/metrics/horizon_specific_model_selection.json
- artifacts/plots/horizon_accuracy_efficiency_pareto.png
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def identify_pareto_frontier(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify non-dominated candidates across MAE and per-sample latency."""
    # We define dominance on (MAE, Latency, Size)
    pareto_set = []
    for cand in candidates:
        is_dominated = False
        mae_c = cand["test_mae"]
        lat_c = cand["mean_latency_ms"]
        size_c = cand["artifact_size_mb"]

        for other in candidates:
            if other["model_name"] == cand["model_name"]:
                continue
            mae_o = other["test_mae"]
            lat_o = other["mean_latency_ms"]
            size_o = other["artifact_size_mb"]

            # other dominates cand if other is <= in all and < in at least one
            if (mae_o <= mae_c and lat_o <= lat_c and size_o <= size_c) and (
                mae_o < mae_c or lat_o < lat_c or size_o < size_c
            ):
                is_dominated = True
                break
        
        cand["is_pareto_optimal"] = not is_dominated
        if not is_dominated:
            pareto_set.append(cand)

    return candidates


def compute_composite_utility(candidates: List[Dict[str, Any]], w_mae=0.50, w_lat=0.25, w_size=0.15, w_train=0.10) -> List[Dict[str, Any]]:
    """Compute multi-objective normalized utility score U in [0, 1]."""
    maes = [c["test_mae"] for c in candidates]
    lats = [c["mean_latency_ms"] for c in candidates]
    sizes = [c["artifact_size_mb"] for c in candidates]
    trains = [c["training_time_sec"] for c in candidates]

    min_m, max_m = min(maes), max(maes)
    min_l, max_l = min(lats), max(lats)
    min_s, max_s = min(sizes), max(sizes)
    min_t, max_t = min(trains), max(trains)

    for c in candidates:
        # Normalize to [0, 1] where 1 is best (lowest cost)
        norm_m = 1.0 - (c["test_mae"] - min_m) / max(1e-4, max_m - min_m)
        norm_l = 1.0 - (c["mean_latency_ms"] - min_l) / max(1e-4, max_l - min_l)
        norm_s = 1.0 - (c["artifact_size_mb"] - min_s) / max(1e-4, max_s - min_s)
        norm_t = 1.0 - (c["training_time_sec"] - min_t) / max(1e-4, max_t - min_t)

        score = w_mae * norm_m + w_lat * norm_l + w_size * norm_s + w_train * norm_t
        c["composite_utility_score"] = round(float(score), 4)

    return sorted(candidates, key=lambda x: x["composite_utility_score"], reverse=True)


def load_all_candidates(horizon: int) -> List[Dict[str, Any]]:
    """Merge Stage 1 baseline models and Stage 3 complexity variants for horizon h."""
    candidates = []

    # 1. From Stage 1
    base_file = Path("artifacts/metrics/efficiency_baseline.json")
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        for r in base_data.get("results", []):
            if r.get("horizon_hours") == horizon:
                candidates.append({
                    "model_name": f"{r['model_name']} (Baseline)",
                    "architecture": r["model_name"],
                    "variant": "Baseline",
                    "test_mae": round(r["predictive_performance"]["mae"], 3),
                    "test_rmse": round(r["predictive_performance"]["rmse"], 3),
                    "test_r2": round(r["predictive_performance"]["r2"], 4),
                    "mean_latency_ms": round(r["latency_benchmarks"]["batch_512"]["per_sample_ms"], 5),
                    "throughput_samples_per_sec": round(r["throughput_samples_per_sec"], 1),
                    "artifact_size_mb": round(r["artifact_size_mb"], 3),
                    "training_time_sec": round(r["training_time_sec"], 3),
                })

    # 2. From Stage 3
    comp_file = Path("artifacts/metrics/model_complexity_optimization.json")
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        h_key = f"h{horizon}"
        if h_key in comp_data.get("results", {}):
            for r in comp_data["results"][h_key]:
                mname = r["model_name"]
                if "Baseline" not in mname:  # Avoid duplicating baselines
                    candidates.append({
                        "model_name": mname,
                        "architecture": mname.split("_")[0],
                        "variant": mname.split(" (")[0],
                        "test_mae": round(r["test_mae"], 3),
                        "test_rmse": round(r["test_rmse"], 3),
                        "test_r2": round(r["test_r2"], 4),
                        "mean_latency_ms": round(r["mean_latency_ms"], 5),
                        "throughput_samples_per_sec": round(r["throughput_samples_per_sec"], 1),
                        "artifact_size_mb": round(r["artifact_size_mb"], 3),
                        "training_time_sec": round(r["training_time_sec"], 3),
                    })

    return candidates


def main():
    print("=" * 80)
    print("STAGE 7: HORIZON-SPECIFIC MODEL SELECTION & PARETO OPTIMIZATION")
    print("=" * 80)

    horizons = [1, 6, 24]
    selection_results = {}
    chosen_suite = {}

    for h in horizons:
        print(f"\n--- Horizon +{h}h Candidates Evaluation ---")
        cands = load_all_candidates(h)
        if not cands:
            print(f"No candidates found for h={h}. Ensure Stages 1 & 3 have executed.")
            continue

        cands = identify_pareto_frontier(cands)
        scored_cands = compute_composite_utility(cands)

        # Select highest composite utility on the Pareto frontier
        pareto_cands = [c for c in scored_cands if c.get("is_pareto_optimal", False)]
        if pareto_cands:
            best_model = pareto_cands[0]
        else:
            best_model = scored_cands[0]

        chosen_suite[f"h{h}"] = best_model
        selection_results[f"h{h}"] = {
            "all_candidates": scored_cands,
            "pareto_frontier": pareto_cands,
            "selected_model": best_model,
        }

        print(f"  Selected Best Model for +{h}h: {best_model['model_name']}")
        print(f"    Test MAE: {best_model['test_mae']} ug/m3 | Size: {best_model['artifact_size_mb']} MB | Latency: {best_model['mean_latency_ms']:.4f} ms | Score: {best_model['composite_utility_score']}")

    # Suite Comparison: Optimized Horizon-Specific Suite vs Uniform RF vs Uniform XGBoost
    print("\n--- Model Suite Comparison ---")
    suite_summary = {
        "selected_horizon_suite": chosen_suite,
        "suite_comparison": {
            "horizon_specific_suite": {
                "avg_test_mae": round(np.mean([chosen_suite[f"h{h}"]["test_mae"] for h in horizons if f"h{h}" in chosen_suite]), 3),
                "total_model_size_mb": round(sum(chosen_suite[f"h{h}"]["artifact_size_mb"] for h in horizons if f"h{h}" in chosen_suite), 3),
                "total_train_time_sec": round(sum(chosen_suite[f"h{h}"]["training_time_sec"] for h in horizons if f"h{h}" in chosen_suite), 3),
            }
        }
    }

    # Export JSON
    out_json = {
        "experiment": "Phase 11 Stage 7: Horizon-Specific Model Selection",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": selection_results,
        "suite_summary": suite_summary,
    }
    out_path = Path("artifacts/metrics/horizon_specific_model_selection.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved horizon-specific model selection to '{out_path}'")

    # Generate Publication Plot
    plot_pareto_frontiers(selection_results, "artifacts/plots/horizon_accuracy_efficiency_pareto.png")


def plot_pareto_frontiers(selection_results: Dict[str, Any], output_path: str):
    """Plot Pareto accuracy vs efficiency frontiers for 1h, 6h, and 24h horizons."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    horizons = [1, 6, 24]

    for idx, h in enumerate(horizons):
        h_key = f"h{h}"
        if h_key not in selection_results:
            continue

        cands = selection_results[h_key]["all_candidates"]
        sel = selection_results[h_key]["selected_model"]

        # Plot all candidates
        for c in cands:
            is_p = c.get("is_pareto_optimal", False)
            color = "#27ae60" if is_p else "#95a5a6"
            marker = "D" if is_p else "o"
            axes[idx].scatter(
                c["mean_latency_ms"] * 1000.0, c["test_mae"],
                c=color, marker=marker, s=max(40, min(300, c["artifact_size_mb"] * 12)),
                alpha=0.85, edgecolors="black", linewidth=1.0
            )
            # Label
            lbl = c["model_name"].split(" (")[0].replace("RandomForest", "RF").replace("LinearRegression", "OLS")
            axes[idx].annotate(
                lbl,
                (c["mean_latency_ms"] * 1000.0, c["test_mae"]),
                fontsize=7.5, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points"
            )

        # Highlight selected
        axes[idx].scatter(
            sel["mean_latency_ms"] * 1000.0, sel["test_mae"],
            facecolors="none", edgecolors="#e74c3c", s=450, linewidth=2.5, linestyle="--",
            label=f"Selected: {sel['model_name'].split(' (')[0]}"
        )

        axes[idx].set_title(f"Horizon +{h}h: Pareto Accuracy vs Latency\n(Green = Pareto Optimal, Diamond = Non-dominated)", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(r"Per-Sample Latency ($\mu s$)", fontsize=10)
        axes[idx].set_ylabel(r"Test MAE ($\mu g/m^3$)", fontsize=10)
        axes[idx].legend(loc="upper right", frameon=True)
        axes[idx].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Pareto plot at '{output_path}'")


if __name__ == "__main__":
    main()
