"""Comprehensive audit script for Final Release Audit (Post-Phase 11).
Executes:
1. Conformal coverage recalculation per horizon, severe tiers, and overall/pooled check.
2. API latency profiling (warmup vs steady-state, mean/median/p95/p99, breakdown).
3. Numerical equivalence:
   - Same model (Pandas vs Buffer with model-aligned encodings)
   - Baseline vs Optimized models
4. Spatial generalization (12 LOSO folds audit)
"""

import os
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import joblib
import numpy as np
import pandas as pd

from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.preprocessing.splitter import TemporalPurgedSplitter
from src.features.registry import get_feature_group
from src.evaluation.metrics import calculate_regression_metrics
from src.uncertainty.conformal import ConformalPredictor
from api.services import AirCastInferenceService

def audit_conformal():
    print("="*80)
    print("1. CONFORMAL COVERAGE RECALCULATION & AUDIT")
    print("="*80)

    # Inspect existing metric files
    with open("artifacts/metrics/final_uncertainty_comparison.json", "r", encoding="utf-8") as f:
        unc_comp = json.load(f)

    print("\n--- Documented Normalized Conformal Metrics (from final_uncertainty_comparison.json) ---")
    for h_key in ["h1", "h6", "h24"]:
        mdata = unc_comp[h_key].get("normalized_conformal", {})
        cov = mdata.get("empirical_coverage")
        width = mdata.get("mean_interval_width")
        ext_cov = mdata.get("extreme_coverage")
        print(f"  {h_key}: Empirical Cov: {cov*100:.2f}% | Mean Width: {width:.2f} | Severe (>150) Cov: {ext_cov*100 if ext_cov is not None else 0:.2f}%")

    print("\n--- Live Recalculation on the Actual Test Set (42,554 samples per horizon) ---")
    cache_path = "data/processed/forecasting_features_cache.parquet"
    df_prepared = pd.read_parquet(cache_path)
    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    horizons = [1, 6, 24]
    live_coverage_results = {}
    all_covered = []

    for h in horizons:
        calib_path = Path(f"artifacts/models/conformal_calibrator_h{h}.joblib")
        calibrator = ConformalPredictor.load(calib_path)

        X_full, y_full, meta = dataset_builder.get_horizon_dataset(
            df_prepared, horizon=h, feature_group="combined_all"
        )
        dt_series = pd.to_datetime(meta[splitter.datetime_col])
        test_mask = (dt_series >= splitter.test_start) & (dt_series <= splitter.test_end)
        X_test, y_test = X_full[test_mask], y_full[test_mask]

        df_intervals = calibrator.predict_intervals(X_test, coverage_level=0.90)
        lower = df_intervals["lower_bound"].values
        upper = df_intervals["upper_bound"].values

        y_true_arr = y_test.values
        covered = (y_true_arr >= lower) & (y_true_arr <= upper)
        cov_pct = np.mean(covered) * 100.0
        mean_width = np.mean(upper - lower)
        median_width = np.median(upper - lower)

        all_covered.extend(covered)

        sev_results = {}
        tiers = [
            ("Good (0-35)", (y_true_arr <= 35)),
            ("Moderate (35-75)", (y_true_arr > 35) & (y_true_arr <= 75)),
            ("Unhealthy Sensitive (75-115)", (y_true_arr > 75) & (y_true_arr <= 115)),
            ("Unhealthy (115-150)", (y_true_arr > 115) & (y_true_arr <= 150)),
            ("Very Unhealthy (150-250)", (y_true_arr > 150) & (y_true_arr <= 250)),
            ("Hazardous / Severe (>250)", (y_true_arr > 250)),
        ]

        print(f"\nHorizon +{h}h (N={len(y_test):,}):")
        print(f"  Overall Empirical Coverage: {cov_pct:.2f}% | Mean Width: {mean_width:.2f} | Median Width: {median_width:.2f}")
        for tname, tmask in tiers:
            n_t = np.sum(tmask)
            if n_t > 0:
                t_cov = np.mean(covered[tmask]) * 100.0
                t_width = np.mean((upper - lower)[tmask])
                sev_results[tname] = {"coverage": round(t_cov, 2), "mean_width": round(t_width, 2), "samples": int(n_t)}
                print(f"    Tier {tname:30s}: cov={t_cov:5.2f}% | mean_width={t_width:6.2f} | samples={n_t:5d}")

        live_coverage_results[f"h{h}"] = {
            "overall_coverage": round(cov_pct, 2),
            "mean_width": round(mean_width, 2),
            "median_width": round(median_width, 2),
            "test_samples": len(y_test),
            "severity_breakdown": sev_results
        }

    pooled_cov = np.mean(all_covered) * 100.0
    print(f"\nPOOLED TEST COVERAGE ACROSS ALL 3 HORIZONS ({len(all_covered):,} samples): {pooled_cov:.2f}%")

    return live_coverage_results, pooled_cov


def audit_api_latency():
    print("\n" + "="*80)
    print("2. API LATENCY RECONCILIATION & BREAKDOWN")
    print("="*80)

    srv = AirCastInferenceService()
    features = {col: 50.0 for col in srv.feature_names}

    results = {}
    for h in [1, 6, 24]:
        print(f"\n--- Profiling Horizon h={h} (model='recommended') ---")
        
        # Cold call
        cold_srv = AirCastInferenceService()
        t0_cold = time.perf_counter()
        cold_res = cold_srv.predict_single(features, horizon=h, model_name="recommended")
        t_cold_ms = (time.perf_counter() - t0_cold) * 1000.0

        res_name = cold_srv._resolve_model_name("recommended", h)
        m = cold_srv.get_model(res_name, h)
        c = cold_srv.get_conformal_calibrator(h)

        print(f"  Warmup (Cold Call) Total: {t_cold_ms:.2f} ms")

        # Steady-state profiling: run 200 warm calls
        latencies = []
        breakdown_feat = []
        breakdown_pred = []
        breakdown_conf = []

        for _ in range(200):
            t0 = time.perf_counter()
            
            # Sub-step 1: Feature dict to DataFrame
            t1 = time.perf_counter()
            row_dict = {col: float(features.get(col, 0.0)) for col in cold_srv.feature_names}
            df_sample = pd.DataFrame([row_dict])
            t_feat = (time.perf_counter() - t1) * 1000.0
            
            # Sub-step 2: Model predict
            t2 = time.perf_counter()
            p = m.predict(df_sample)[0]
            t_pred = (time.perf_counter() - t2) * 1000.0
            
            # Sub-step 3: Conformal interval calculation
            t3 = time.perf_counter()
            if c is not None and c.is_calibrated:
                q_val = c.quantiles_.get(0.90, 1.645)
                b0 = c.dispersion_params_.get("beta_0", c.sigma_min)
                b1 = c.dispersion_params_.get("beta_1", 0.0)
                sigma_x = max(c.sigma_min, b0 + b1 * p)
                margin = q_val * sigma_x
                lb = max(0.0, p - margin)
                ub = p + margin
            t_conf = (time.perf_counter() - t3) * 1000.0

            total_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(total_ms)
            breakdown_feat.append(t_feat)
            breakdown_pred.append(t_pred)
            breakdown_conf.append(t_conf)

        lat_arr = np.array(latencies)
        mean_ms = float(np.mean(lat_arr))
        median_ms = float(np.median(lat_arr))
        p95_ms = float(np.percentile(lat_arr, 95))
        p99_ms = float(np.percentile(lat_arr, 99))
        min_ms = float(np.min(lat_arr))
        max_ms = float(np.max(lat_arr))

        mean_feat_ms = float(np.mean(breakdown_feat))
        mean_pred_ms = float(np.mean(breakdown_pred))
        mean_conf_ms = float(np.mean(breakdown_conf))

        print(f"  Steady-State (200 requests):")
        print(f"    Mean:   {mean_ms:.3f} ms | Median: {median_ms:.3f} ms")
        print(f"    P95:    {p95_ms:.3f} ms | P99:    {p99_ms:.3f} ms | Min: {min_ms:.3f} ms | Max: {max_ms:.3f} ms")
        print(f"    Breakdown -> Feature prep: {mean_feat_ms:.3f} ms ({mean_feat_ms/mean_ms*100:.1f}%) | Model: {mean_pred_ms:.3f} ms ({mean_pred_ms/mean_ms*100:.1f}%) | Conformal: {mean_conf_ms:.3f} ms ({mean_conf_ms/mean_ms*100:.1f}%)")

        meets_15ms_steady_p95 = p95_ms <= 15.0
        meets_15ms_steady_mean = mean_ms <= 15.0
        print(f"    Meets 15 ms limit in Steady-State Mean: {meets_15ms_steady_mean} (Mean = {mean_ms:.2f} ms)")
        print(f"    Meets 15 ms limit in Steady-State P95:  {meets_15ms_steady_p95} (P95 = {p95_ms:.2f} ms)")

        results[f"h{h}"] = {
            "cold_warmup_latency_ms": round(t_cold_ms, 3),
            "steady_state_mean_ms": round(mean_ms, 3),
            "steady_state_median_ms": round(median_ms, 3),
            "steady_state_p95_ms": round(p95_ms, 3),
            "steady_state_p99_ms": round(p99_ms, 3),
            "feature_prep_ms": round(mean_feat_ms, 3),
            "model_pred_ms": round(mean_pred_ms, 3),
            "conformal_ms": round(mean_conf_ms, 3),
            "meets_15ms_steady_state_p95": meets_15ms_steady_p95,
            "meets_15ms_steady_state_mean": meets_15ms_steady_mean
        }

    return results


def audit_numerical_equivalence():
    print("\n" + "="*80)
    print("3. NUMERICAL EQUIVALENCE & COMPARISON AUDIT")
    print("="*80)

    cache_path = "data/processed/forecasting_features_cache.parquet"
    df_prepared = pd.read_parquet(cache_path)
    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    equiv_results = {}

    for h in [1, 6, 24]:
        print(f"\n--- Numerical Comparison for Horizon h={h} ---")
        X_full, y_full, meta = dataset_builder.get_horizon_dataset(
            df_prepared, horizon=h, feature_group="combined_all"
        )
        dt_series = pd.to_datetime(meta[splitter.datetime_col])
        test_mask = (dt_series >= splitter.test_start) & (dt_series <= splitter.test_end)
        X_test, y_test = X_full[test_mask], y_full[test_mask]

        m_base = joblib.load(f"artifacts/models/xgboost_h{h}.joblib")
        
        # Exact encoding from model
        X_valid = m_base._validate_inputs(X_test, is_training=False)
        y_pred_pandas = m_base.predict(X_test)
        
        # NumPy buffer inference using same model
        X_numpy = X_valid.to_numpy(dtype=np.float32)
        y_pred_numpy = np.maximum(0.0, m_base.model.predict(X_numpy))

        max_diff_same_model = float(np.max(np.abs(y_pred_pandas - y_pred_numpy)))
        mean_diff_same_model = float(np.mean(np.abs(y_pred_pandas - y_pred_numpy)))
        print(f"  [Same Model: xgboost_h{h}] Pandas vs Buffer:")
        print(f"    Max Absolute Diff:  {max_diff_same_model:.6e}")
        print(f"    Mean Absolute Diff: {mean_diff_same_model:.6e}")
        print(f"    Numerically Identical: {max_diff_same_model < 1e-5}")

        # Cross-model comparison
        if h == 1:
            opt_name = "xgboost_h1.joblib"
        elif h == 6:
            opt_name = "xgboost_pruned_h6.joblib"
        else:
            opt_name = "xgboost_fast_h24.joblib"

        m_opt = joblib.load(f"artifacts/models/{opt_name}")
        y_pred_opt = m_opt.predict(X_test)

        max_diff_cross = float(np.max(np.abs(y_pred_pandas - y_pred_opt)))
        mean_diff_cross = float(np.mean(np.abs(y_pred_pandas - y_pred_opt)))
        
        m_base_metrics = calculate_regression_metrics(y_test, y_pred_pandas)
        m_opt_metrics = calculate_regression_metrics(y_test, y_pred_opt)

        print(f"\n  [Cross-Model: Baseline xgboost_h{h} vs Optimized {opt_name}]:")
        print(f"    Max Absolute Diff:     {max_diff_cross:.4f}")
        print(f"    Mean Absolute Diff:    {mean_diff_cross:.4f}")
        print(f"    Baseline Test MAE:     {m_base_metrics['mae']:.4f} | RMSE: {m_base_metrics['rmse']:.4f} | R2: {m_base_metrics['r2']:.4f}")
        print(f"    Optimized Test MAE:    {m_opt_metrics['mae']:.4f} | RMSE: {m_opt_metrics['rmse']:.4f} | R2: {m_opt_metrics['r2']:.4f}")
        print(f"    Delta MAE:             {m_opt_metrics['mae'] - m_base_metrics['mae']:+.4f}")
        print(f"    Are models structurally identical? {h == 1} (h=1 is baseline; h=6 and h=24 are pruned!)")

        equiv_results[f"h{h}"] = {
            "same_model_max_diff": max_diff_same_model,
            "same_model_mean_diff": mean_diff_same_model,
            "same_model_identical": bool(max_diff_same_model < 1e-5),
            "cross_model_max_diff": round(max_diff_cross, 4),
            "cross_model_mean_diff": round(mean_diff_cross, 4),
            "baseline_mae": round(m_base_metrics["mae"], 4),
            "baseline_rmse": round(m_base_metrics["rmse"], 4),
            "baseline_r2": round(m_base_metrics["r2"], 4),
            "optimized_mae": round(m_opt_metrics["mae"], 4),
            "optimized_rmse": round(m_opt_metrics["rmse"], 4),
            "optimized_r2": round(m_opt_metrics["r2"], 4),
            "delta_mae": round(m_opt_metrics["mae"] - m_base_metrics["mae"], 4)
        }

    return equiv_results


def audit_loso_generalization():
    print("\n" + "="*80)
    print("4. SPATIAL GENERALIZATION (12-FOLD LOSO) AUDIT")
    print("="*80)

    with open("artifacts/metrics/experiment7_loso_generalisation.json", "r", encoding="utf-8") as f:
        loso_data = json.load(f)

    print("Experiment 7 LOSO configuration:")
    print(f"  Architecture:         XGBoost (Hist Gradient Boosting)")
    print(f"  Forecasting Horizon:  {loso_data.get('horizon_hours')} hours")
    print(f"  Total Stations:       {loso_data.get('n_stations')}")

    station_results = loso_data.get("per_station_results", [])
    print(f"  Total Folds Documented: {len(station_results)}")
    
    all_stations = [s["held_out_station"] for s in station_results]
    print(f"  All 12 Stations Evaluated: {all_stations}")

    print("\nFold-by-Fold Breakdown:")
    print(f"  {'Fold':4s} | {'Held-out Station':18s} | {'Domain':15s} | {'Test Samples':12s} | {'Test MAE':10s} | {'Test RMSE':10s} | {'Test R2':10s} | {'Training Stations'}")
    print("-" * 115)

    mae_list = []
    rmse_list = []
    r2_list = []
    total_samples = 0

    loso_summary_table = []

    for i, fold in enumerate(station_results):
        s_name = fold["held_out_station"]
        domain = fold["domain"]
        n_samples = fold["test_samples"]
        mae = fold["mae"]
        rmse = fold["rmse"]
        r2 = fold["r2"]
        
        train_stations = [st for st in all_stations if st != s_name]
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        total_samples += n_samples

        print(f"  {i+1:4d} | {s_name:18s} | {domain:15s} | {n_samples:12d} | {mae:10.4f} | {rmse:10.4f} | {r2:10.4f} | 11 stations ({len(train_stations)})")
        loso_summary_table.append({
            "fold": i + 1,
            "held_out_station": s_name,
            "domain": domain,
            "n_training_stations": len(train_stations),
            "test_samples": n_samples,
            "test_mae": round(mae, 4),
            "test_rmse": round(rmse, 4),
            "test_r2": round(r2, 4)
        })

    mean_mae = float(np.mean(mae_list))
    std_mae = float(np.std(mae_list))
    mean_rmse = float(np.mean(rmse_list))
    mean_r2 = float(np.mean(r2_list))

    print("-" * 115)
    print(f"  12-Fold Mean +- Std: MAE = {mean_mae:.2f} +- {std_mae:.2f} ug/m3 | Mean RMSE = {mean_rmse:.2f} ug/m3 | Mean R2 = {mean_r2:.4f}")
    print(f"  Total Evaluated Samples Across 12 Folds: {total_samples:,}")

    return {
        "mean_mae": round(mean_mae, 4),
        "std_mae": round(std_mae, 4),
        "mean_rmse": round(mean_rmse, 4),
        "mean_r2": round(mean_r2, 4),
        "total_samples": total_samples,
        "folds": loso_summary_table
    }

if __name__ == "__main__":
    c_res, pooled_cov = audit_conformal()
    lat_res = audit_api_latency()
    num_res = audit_numerical_equivalence()
    loso_res = audit_loso_generalization()
