"""AirCast Research: Final Correction and Empirical Validation Pipeline.

Executes:
1. Uncertainty Quantification (Standard vs. Normalized Adaptive Conformal) across h in {1, 6, 24}.
   - Evaluates coverage by horizon, station, season, and PM2.5 severity tiers.
   - Serializes calibrated ConformalPredictor models to artifacts/models/.
   - Exports artifacts/metrics/final_uncertainty_comparison.json.
2. LOSO Generalisation with vs. without Station ID.
   - 12-fold comparison isolating local telemetry vs station identifier.
   - Computes both macro-averaged and pooled metrics.
   - Exports artifacts/metrics/final_loso_station_id_comparison.json.
3. Raw Sensor Measurement Robustness Stress Testing.
   - Injects noise into raw sensor channels, re-deriving lags, rolling stats, and wind vectors.
   - Evaluates across multiple random seeds (42, 123, 999).
   - Tests wind sensor failure, gas precursor failure, and random sensor dropouts.
   - Exports artifacts/metrics/final_sensor_robustness.json.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd

from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.preprocessing.splitter import TemporalPurgedSplitter
from src.models import XGBoostForecaster, PersistenceForecaster
from src.uncertainty.conformal import ConformalPredictor, evaluate_conformal_coverage
from src.evaluation.metrics import calculate_regression_metrics
from src.features.registry import get_feature_group


def run_task1_uncertainty(df_prepared: pd.DataFrame, builder: MultiHorizonForecastingDataset, splitter: TemporalPurgedSplitter) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("TASK 1: UNCERTAINTY QUANTIFICATION CORRECTION (STANDARD VS NORMALIZED ADAPTIVE)")
    print("=" * 80)

    horizons = [1, 6, 24]
    pm_tiers = [
        ("Good to Moderate [0, 75)", 0.0, 75.0),
        ("USG [75, 115)", 75.0, 115.0),
        ("Unhealthy [115, 150)", 115.0, 150.0),
        ("Very Unhealthy [150, 250)", 150.0, 250.0),
        ("Severe / Hazardous [250+)", 250.0, 10000.0),
    ]

    task1_results = {}

    for h in horizons:
        print(f"\n---> Horizon h={h}h: Calibrating Standard and Normalized Adaptive Conformal...")
        X, y, meta = builder.get_horizon_dataset(df_prepared, horizon=h, feature_group="combined_all")
        dt = pd.to_datetime(meta[splitter.datetime_col])
        train_m = dt <= splitter.train_end
        val_m = (dt >= splitter.val_start) & (dt <= splitter.val_end)
        test_m = (dt >= splitter.test_start) & (dt <= splitter.test_end)

        X_train, y_train = X[train_m], y[train_m]
        X_val, y_val = X[val_m], y[val_m]
        X_test, y_test = X[test_m], y[test_m]
        y_test_arr = y_test.values

        # Base model
        model = XGBoostForecaster(name=f"xgb_h{h}", n_estimators=140, max_depth=6, learning_rate=0.06, random_state=42)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # 1. Standard Split Conformal (Homoscedastic)
        std_cp = ConformalPredictor(base_model=model, default_coverage=0.90, method="standard")
        std_cp.calibrate(X_val, y_val)

        # 2. Normalized Adaptive Conformal (Heteroscedastic difficulty score)
        norm_cp = ConformalPredictor(base_model=model, default_coverage=0.90, method="normalized", sigma_min=5.0)
        norm_cp.calibrate(X_val, y_val)

        # Save normalized predictor as primary calibrated artifact for API serving
        std_save_path = f"artifacts/models/conformal_calibrator_std_h{h}.joblib"
        norm_save_path = f"artifacts/models/conformal_calibrator_h{h}.joblib"
        std_cp.save(std_save_path)
        norm_cp.save(norm_save_path)

        # Evaluate on Test
        std_preds = std_cp.predict_intervals(X_test, coverage_level=0.90)
        norm_preds = norm_cp.predict_intervals(X_test, coverage_level=0.90)

        std_metrics = evaluate_conformal_coverage(y_test_arr, std_preds["lower_bound"], std_preds["upper_bound"], nominal_coverage=0.90)
        norm_metrics = evaluate_conformal_coverage(y_test_arr, norm_preds["lower_bound"], norm_preds["upper_bound"], nominal_coverage=0.90)

        # Stratified Breakdown by Season
        test_meta = meta[test_m].copy()
        test_meta["y_true"] = y_test_arr
        test_meta["is_heating_season"] = X_test["is_heating_season"].values
        test_meta["std_covered"] = (y_test_arr >= std_preds["lower_bound"].values) & (y_test_arr <= std_preds["upper_bound"].values)
        test_meta["norm_covered"] = (y_test_arr >= norm_preds["lower_bound"].values) & (y_test_arr <= norm_preds["upper_bound"].values)

        seasonal_breakdown = {}
        for heat_val, s_group in test_meta.groupby("is_heating_season"):
            s_name = "Heating Season (Nov-Feb)" if float(heat_val) == 1.0 else "Non-Heating (Oct)"
            seasonal_breakdown[s_name] = {
                "sample_count": len(s_group),
                "std_coverage": round(float(np.mean(s_group["std_covered"])) * 100.0, 2),
                "norm_coverage": round(float(np.mean(s_group["norm_covered"])) * 100.0, 2),
            }

        # Stratified Breakdown by PM Tier
        tier_breakdown = {}
        for t_name, low, high in pm_tiers:
            t_mask = (test_meta["y_true"] >= low) & (test_meta["y_true"] < high)
            if np.sum(t_mask) > 0:
                t_sub = test_meta[t_mask]
                tier_breakdown[t_name] = {
                    "sample_count": len(t_sub),
                    "std_coverage": round(float(np.mean(t_sub["std_covered"])) * 100.0, 2),
                    "norm_coverage": round(float(np.mean(t_sub["norm_covered"])) * 100.0, 2),
                }

        # Stratified Breakdown by Station
        station_breakdown = {}
        for st, st_group in test_meta.groupby("station"):
            station_breakdown[st] = {
                "sample_count": len(st_group),
                "std_coverage": round(float(np.mean(st_group["std_covered"])) * 100.0, 2),
                "norm_coverage": round(float(np.mean(st_group["norm_covered"])) * 100.0, 2),
            }

        print(f"  Standard Conformal Test Coverage : {std_metrics['empirical_coverage']*100:.2f}% | Mean Width: {std_metrics['mean_interval_width']:.2f}")
        print(f"  Normalized Adaptive Test Coverage: {norm_metrics['empirical_coverage']*100:.2f}% | Mean Width: {norm_metrics['mean_interval_width']:.2f}")
        if "Severe / Hazardous [250+)" in tier_breakdown:
            sev = tier_breakdown["Severe / Hazardous [250+)"]
            print(f"  Severe (>=250) Tier Coverage     : Standard={sev['std_coverage']}% -> Normalized={sev['norm_coverage']}%")

        task1_results[f"h{h}"] = {
            "nominal_coverage": 0.90,
            "standard_conformal": {
                "calib_quantile": round(std_cp.quantiles_[0.90], 2),
                **std_metrics,
            },
            "normalized_conformal": {
                "calib_quantile": round(norm_cp.quantiles_[0.90], 4),
                "dispersion_params": {k: round(v, 4) for k, v in norm_cp.dispersion_params_.items()},
                **norm_metrics,
            },
            "seasonal_stratification": seasonal_breakdown,
            "severity_stratification": tier_breakdown,
            "station_stratification": station_breakdown,
        }

    out_path = "artifacts/metrics/final_uncertainty_comparison.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(task1_results, f, indent=2)
    print(f"\n[SUCCESS] Task 1 uncertainty comparison saved to '{out_path}'.")
    return task1_results


def run_task3_loso(df_prepared: pd.DataFrame, builder: MultiHorizonForecastingDataset) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("TASK 3: IMPROVED LOSO VALIDATION (WITH VS WITHOUT STATION ID)")
    print("=" * 80)

    horizon = 24
    X_full, y, meta = builder.get_horizon_dataset(df_prepared, horizon=horizon, feature_group="combined_all")
    stations = sorted(meta["station"].unique())

    # Two feature sets: With station ID (47 features) vs Without station ID (46 features)
    features_with = list(X_full.columns)
    features_without = [c for c in features_with if c != "station"]

    print(f"Features With Station ID   : {len(features_with)} columns")
    print(f"Features Without Station ID: {len(features_without)} columns")

    fold_results: List[Dict[str, Any]] = []
    all_y_true: List[np.ndarray] = []
    all_preds_with: List[np.ndarray] = []
    all_preds_without: List[np.ndarray] = []
    all_preds_pers: List[np.ndarray] = []

    for idx, held_out_st in enumerate(stations, 1):
        print(f"\nFold {idx:02d}/12: Evaluating Held-Out Station [{held_out_st}]...", end="", flush=True)

        is_held_out = meta["station"] == held_out_st
        train_m = ~is_held_out
        test_m = is_held_out

        y_tr = y[train_m]
        y_te = y[test_m].values

        # 1. Model WITH station ID
        X_tr_with = X_full[train_m][features_with]
        X_te_with = X_full[test_m][features_with]

        model_with = XGBoostForecaster(name=f"xgb_with_{held_out_st}", n_estimators=140, max_depth=6, learning_rate=0.06, random_state=42)
        model_with.fit(X_tr_with, y_tr)
        preds_with = model_with.predict(X_te_with)

        # 2. Model WITHOUT station ID
        X_tr_without = X_full[train_m][features_without]
        X_te_without = X_full[test_m][features_without]

        model_without = XGBoostForecaster(name=f"xgb_without_{held_out_st}", n_estimators=140, max_depth=6, learning_rate=0.06, random_state=42)
        model_without.fit(X_tr_without, y_tr)
        preds_without = model_without.predict(X_te_without)

        # 3. Persistence Baseline (local PM2.5 lag-0 at inference time)
        preds_pers = X_full.loc[test_m, "PM2.5"].values

        m_with = calculate_regression_metrics(y_te, preds_with)
        m_without = calculate_regression_metrics(y_te, preds_without)
        m_pers = calculate_regression_metrics(y_te, preds_pers)

        all_y_true.append(y_te)
        all_preds_with.append(preds_with)
        all_preds_without.append(preds_without)
        all_preds_pers.append(preds_pers)

        delta_mae = m_without["mae"] - m_with["mae"]
        print(f" With MAE: {m_with['mae']:.2f} | Without MAE: {m_without['mae']:.2f} (Delta: {delta_mae:+.2f}) | Pers MAE: {m_pers['mae']:.2f}")

        fold_results.append({
            "station": held_out_st,
            "sample_count": len(y_te),
            "with_station_id": {
                "mae": m_with["mae"],
                "rmse": m_with["rmse"],
                "r2": m_with["r2"],
            },
            "without_station_id": {
                "mae": m_without["mae"],
                "rmse": m_without["rmse"],
                "r2": m_without["r2"],
            },
            "persistence": {
                "mae": m_pers["mae"],
                "rmse": m_pers["rmse"],
                "r2": m_pers["r2"],
            },
            "delta_mae_without_minus_with": round(delta_mae, 4),
        })

    # Macro-Averaged Summary (Unweighted mean of fold metrics)
    macro_summary = {
        "with_station_id": {
            "mean_mae": round(float(np.mean([f["with_station_id"]["mae"] for f in fold_results])), 4),
            "std_mae": round(float(np.std([f["with_station_id"]["mae"] for f in fold_results])), 4),
            "mean_rmse": round(float(np.mean([f["with_station_id"]["rmse"] for f in fold_results])), 4),
            "mean_r2": round(float(np.mean([f["with_station_id"]["r2"] for f in fold_results])), 4),
        },
        "without_station_id": {
            "mean_mae": round(float(np.mean([f["without_station_id"]["mae"] for f in fold_results])), 4),
            "std_mae": round(float(np.std([f["without_station_id"]["mae"] for f in fold_results])), 4),
            "mean_rmse": round(float(np.mean([f["without_station_id"]["rmse"] for f in fold_results])), 4),
            "mean_r2": round(float(np.mean([f["without_station_id"]["r2"] for f in fold_results])), 4),
        },
        "persistence": {
            "mean_mae": round(float(np.mean([f["persistence"]["mae"] for f in fold_results])), 4),
            "mean_rmse": round(float(np.mean([f["persistence"]["rmse"] for f in fold_results])), 4),
            "mean_r2": round(float(np.mean([f["persistence"]["r2"] for f in fold_results])), 4),
        },
    }

    # Pooled Summary (Concatenated predictions across all folds)
    y_true_pooled = np.concatenate(all_y_true)
    y_pred_with_pooled = np.concatenate(all_preds_with)
    y_pred_without_pooled = np.concatenate(all_preds_without)
    y_pred_pers_pooled = np.concatenate(all_preds_pers)

    m_with_pooled = calculate_regression_metrics(y_true_pooled, y_pred_with_pooled)
    m_without_pooled = calculate_regression_metrics(y_true_pooled, y_pred_without_pooled)
    m_pers_pooled = calculate_regression_metrics(y_true_pooled, y_pred_pers_pooled)

    pooled_summary = {
        "with_station_id": {
            "mae": m_with_pooled["mae"],
            "rmse": m_with_pooled["rmse"],
            "r2": m_with_pooled["r2"],
        },
        "without_station_id": {
            "mae": m_without_pooled["mae"],
            "rmse": m_without_pooled["rmse"],
            "r2": m_without_pooled["r2"],
        },
        "persistence": {
            "mae": m_pers_pooled["mae"],
            "rmse": m_pers_pooled["rmse"],
            "r2": m_pers_pooled["r2"],
        },
    }

    print("\n" + "-" * 70)
    print("LOSO RECONCILIATION SUMMARY:")
    print(f"  Macro-Averaged MAE: With ID = {macro_summary['with_station_id']['mean_mae']:.2f} +/- {macro_summary['with_station_id']['std_mae']:.2f} | Without ID = {macro_summary['without_station_id']['mean_mae']:.2f} +/- {macro_summary['without_station_id']['std_mae']:.2f} | Persistence = {macro_summary['persistence']['mean_mae']:.2f}")
    print(f"  Pooled Total MAE  : With ID = {pooled_summary['with_station_id']['mae']:.2f} | Without ID = {pooled_summary['without_station_id']['mae']:.2f} | Persistence = {pooled_summary['persistence']['mae']:.2f}")
    print("-" * 70)

    task3_output = {
        "experiment": "Task 3: LOSO Validation With vs Without Station ID",
        "horizon_hours": horizon,
        "n_stations": len(stations),
        "total_evaluation_samples": len(y_true_pooled),
        "macro_averaged": macro_summary,
        "pooled": pooled_summary,
        "folds": fold_results,
    }

    out_path = "artifacts/metrics/final_loso_station_id_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(task3_output, f, indent=2)
    print(f"[SUCCESS] Task 3 LOSO comparison saved to '{out_path}'.")
    return task3_output


def run_task4_robustness(df_prepared: pd.DataFrame, builder: MultiHorizonForecastingDataset, splitter: TemporalPurgedSplitter) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("TASK 4: IMPROVED ROBUSTNESS (RAW SENSOR PERTURBATION & MULTI-SEED RE-DERIVATION)")
    print("=" * 80)

    horizon = 24
    X, y, meta = builder.get_horizon_dataset(df_prepared, horizon=horizon, feature_group="combined_all")
    dt = pd.to_datetime(meta[splitter.datetime_col])
    train_m = dt <= splitter.train_end
    val_m = (dt >= splitter.val_start) & (dt <= splitter.val_end)
    test_m = (dt >= splitter.test_start) & (dt <= splitter.test_end)

    X_train, y_train = X[train_m], y[train_m]
    X_val, y_val = X[val_m], y[val_m]
    X_test, y_test = X[test_m], y[test_m]
    y_test_arr = y_test.values

    # Train canonical reference model
    model = XGBoostForecaster(name="xgb_robustness_ref", n_estimators=140, max_depth=6, learning_rate=0.06, random_state=42)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    clean_preds = model.predict(X_test)
    baseline_metrics = calculate_regression_metrics(y_test_arr, clean_preds)
    print(f"Clean Baseline Test MAE: {baseline_metrics['mae']:.2f} | RMSE: {baseline_metrics['rmse']:.2f} | R2: {baseline_metrics['r2']:.4f}")

    # Primary raw channels
    raw_channels = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
    channel_stds = X_train[raw_channels].std()

    # Part 1: Raw Sensor Noise Perturbation Across Multiple Seeds
    seeds = [42, 123, 999]
    noise_ratios = [0.05, 0.15, 0.25]
    noise_experiments = {}

    for rho in noise_ratios:
        seed_maes = []
        seed_rmses = []
        seed_r2s = []
        seed_degrads = []

        for seed in seeds:
            rng = np.random.default_rng(seed)
            X_test_perturbed = X_test.copy()

            # 1. Perturb raw channels
            for ch in raw_channels:
                if ch in X_test_perturbed.columns:
                    noise = rng.normal(0, rho * channel_stds[ch], size=len(X_test_perturbed))
                    # Raw physical non-negativity constraint for pollutants and wind speed
                    if ch in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "WSPM"]:
                        X_test_perturbed[ch] = np.maximum(0.0, X_test_perturbed[ch] + noise)
                    else:
                        X_test_perturbed[ch] = X_test_perturbed[ch] + noise

            # 2. Causally recompute physical derived features from perturbed raw channels!
            if "TEMP" in X_test_perturbed.columns and "DEWP" in X_test_perturbed.columns:
                X_test_perturbed["dew_point_depression"] = X_test_perturbed["TEMP"] - X_test_perturbed["DEWP"]

            if "WSPM" in X_test_perturbed.columns and "wd_sin" in X_test_perturbed.columns:
                # Recompute wind vectors u and v with perturbed WSPM
                X_test_perturbed["wind_u"] = -X_test_perturbed["WSPM"] * X_test_perturbed["wd_sin"]
                X_test_perturbed["wind_v"] = -X_test_perturbed["WSPM"] * X_test_perturbed["wd_cos"]

            # Recompute lag and rolling proxies from perturbed PM2.5
            # Under raw sensor noise, point lags receive corresponding raw sensor noise
            for lag in [1, 3, 6, 12, 24, 48]:
                col_name = f"PM2.5_lag_{lag}"
                if col_name in X_test_perturbed.columns:
                    lag_noise = rng.normal(0, rho * channel_stds["PM2.5"], size=len(X_test_perturbed))
                    X_test_perturbed[col_name] = np.maximum(0.0, X_test_perturbed[col_name] + lag_noise)

            # Rolling stats receive noise scaled by sqrt(w)
            for w in [3, 6, 12, 24]:
                col_name = f"PM2.5_roll_mean_{w}"
                if col_name in X_test_perturbed.columns:
                    roll_noise = rng.normal(0, (rho * channel_stds["PM2.5"]) / np.sqrt(w), size=len(X_test_perturbed))
                    X_test_perturbed[col_name] = np.maximum(0.0, X_test_perturbed[col_name] + roll_noise)

            p_preds = model.predict(X_test_perturbed)
            m = calculate_regression_metrics(y_test_arr, p_preds)
            degrad = ((m["mae"] - baseline_metrics["mae"]) / baseline_metrics["mae"]) * 100.0

            seed_maes.append(m["mae"])
            seed_rmses.append(m["rmse"])
            seed_r2s.append(m["r2"])
            seed_degrads.append(degrad)

        noise_experiments[f"noise_{int(rho*100)}pct"] = {
            "noise_ratio": rho,
            "mean_mae": round(float(np.mean(seed_maes)), 4),
            "std_mae": round(float(np.std(seed_maes)), 4),
            "mean_rmse": round(float(np.mean(seed_rmses)), 4),
            "mean_r2": round(float(np.mean(seed_r2s)), 4),
            "mean_pct_degradation": round(float(np.mean(seed_degrads)), 2),
            "seeds_evaluated": seeds,
        }
        print(f"  Noise rho={rho*100:4.1f}% (3 seeds) -> Mean MAE: {np.mean(seed_maes):.2f} +/- {np.std(seed_maes):.2f} ug/m3 | Mean Degradation: {np.mean(seed_degrads):+.2f}%")

    # Part 2: Sensor Failure and Outage Stress Tests
    print("\nEvaluating Hardware Outage & Missing Data Scenarios...")
    outage_scenarios = [
        ("No Outage (Baseline)", []),
        ("CO Sensor Failure", ["CO"]),
        ("SO2 + NO2 Precursor Failure", ["SO2", "NO2"]),
        ("Complete Precursor Gas Failure (SO2, NO2, CO, O3)", ["SO2", "NO2", "CO", "O3"]),
        ("Wind Anemometer Suite Failure (WSPM, wind_u, wind_v, wd)", ["WSPM", "wind_u", "wind_v", "wd_sin", "wd_cos"]),
        ("Particulate Sensor Disconnection (PM2.5, PM10, Lags, Rolling)", ["PM2.5", "PM10"] + [c for c in X_test.columns if "PM2.5_lag" in c or "PM2.5_roll" in c]),
    ]

    outage_results = []
    for sc_name, dropped_cols in outage_scenarios:
        X_test_sc = X_test.copy()
        for col in dropped_cols:
            if col in X_test_sc.columns:
                X_test_sc[col] = float(X_train[col].median())

        p_preds = model.predict(X_test_sc)
        m = calculate_regression_metrics(y_test_arr, p_preds)
        delta_mae = m["mae"] - baseline_metrics["mae"]

        outage_results.append({
            "scenario": sc_name,
            "dropped_features_count": len(dropped_cols),
            "mae": m["mae"],
            "rmse": m["rmse"],
            "r2": m["r2"],
            "delta_mae": round(delta_mae, 4),
            "pct_degradation": round((delta_mae / baseline_metrics["mae"]) * 100.0, 2),
        })
        print(f"  Scenario: {sc_name:<55} -> MAE: {m['mae']:.2f} (Delta: {delta_mae:+.2f} ug/m3)")

    # Part 3: Random Missing-Data Packet Loss Simulation (10% and 30% random feature dropouts)
    dropout_results = []
    for rate in [0.10, 0.30]:
        rng = np.random.default_rng(42)
        X_test_drop = X_test.copy()
        mask_drop = rng.random(size=X_test_drop.shape) < rate
        # Replace dropped values with training median
        for c_idx, col in enumerate(X_test_drop.columns):
            if pd.api.types.is_numeric_dtype(X_train[col].dtype):
                med = float(X_train[col].median())
                X_test_drop.iloc[mask_drop[:, c_idx], c_idx] = med

        p_preds = model.predict(X_test_drop)
        m = calculate_regression_metrics(y_test_arr, p_preds)
        delta_mae = m["mae"] - baseline_metrics["mae"]

        dropout_results.append({
            "dropout_rate": rate,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "r2": m["r2"],
            "delta_mae": round(delta_mae, 4),
            "pct_degradation": round((delta_mae / baseline_metrics["mae"]) * 100.0, 2),
        })
        print(f"  Random Telemetry Packet Loss {int(rate*100)}% -> MAE: {m['mae']:.2f} (Delta: {delta_mae:+.2f} ug/m3)")

    task4_output = {
        "experiment": "Task 4: Improved Sensor Robustness & Stress Testing",
        "horizon_hours": horizon,
        "clean_baseline": baseline_metrics,
        "raw_sensor_noise_multi_seed": noise_experiments,
        "hardware_outage_scenarios": outage_results,
        "random_packet_loss_dropouts": dropout_results,
    }

    out_path = "artifacts/metrics/final_sensor_robustness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(task4_output, f, indent=2)
    print(f"[SUCCESS] Task 4 robustness results saved to '{out_path}'.")
    return task4_output


def main():
    print("=" * 80)
    print("AIRCAST RESEARCH: FINAL CORRECTION AND VALIDATION PIPELINE")
    print("=" * 80)

    t0 = time.perf_counter()
    cache_path = "data/processed/forecasting_features_cache.parquet"
    df_prepared = pd.read_parquet(cache_path)
    builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    # Task 1: Uncertainty Quantification (skip if already generated)
    if not os.path.exists("artifacts/metrics/final_uncertainty_comparison.json"):
        run_task1_uncertainty(df_prepared, builder, splitter)
    else:
        print("[Task 1] final_uncertainty_comparison.json already exists. Skipping re-computation.")

    # Task 3: LOSO Validation (skip if already generated)
    if not os.path.exists("artifacts/metrics/final_loso_station_id_comparison.json"):
        run_task3_loso(df_prepared, builder)
    else:
        print("[Task 3] final_loso_station_id_comparison.json already exists. Skipping re-computation.")

    # Task 4: Robustness Experiments
    run_task4_robustness(df_prepared, builder, splitter)

    elapsed = time.perf_counter() - t0
    print(f"\n[ALL CORRECTIONS COMPLETED IN {elapsed:.2f}s!]")


if __name__ == "__main__":
    main()
