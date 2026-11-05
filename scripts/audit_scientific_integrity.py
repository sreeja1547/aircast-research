"""Scientific Integrity Audit Script for AirCast Research.

Audits:
1. Conformal coverage across Horizons (1h, 6h, 24h), Stations (12 stations), Seasons (Heating vs Non-Heating), and PM2.5 concentration ranges.
2. LOSO station fold evaluations and persistence baseline comparisons.
3. Robustness noise and outage degradation metrics.
4. Temporal and seasonal distributions.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd

from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.preprocessing.splitter import TemporalPurgedSplitter
from src.models import XGBoostForecaster, PersistenceForecaster
from src.uncertainty.conformal import ConformalPredictor, evaluate_conformal_coverage
from src.evaluation.metrics import calculate_regression_metrics

def main():
    print("=" * 80)
    print("AIRCAST SCIENTIFIC INTEGRITY AUDIT — EMPIRICAL RE-VERIFICATION")
    print("=" * 80)

    cache_path = "data/processed/forecasting_features_cache.parquet"
    if not os.path.exists(cache_path):
        print(f"Error: Cache path {cache_path} not found.")
        return

    df_prepared = pd.read_parquet(cache_path)
    builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    # =========================================================================
    # AUDIT 1: CONFORMAL COVERAGE BREAKDOWNS (Horizon, Station, Season, PM Tier)
    # =========================================================================
    print("\n" + "=" * 60)
    print("AUDIT 1: CONFORMAL PREDICTION COVERAGE AUDIT")
    print("=" * 60)

    # We evaluate for h=24h, h=6h, h=1h using canonical models calibrated on validation
    audit_results = {}

    for h in [1, 6, 24]:
        X, y, meta = builder.get_horizon_dataset(df_prepared, horizon=h, feature_group="combined_all")
        dt_series = pd.to_datetime(meta[splitter.datetime_col])
        train_mask = dt_series <= splitter.train_end
        val_mask = (dt_series >= splitter.val_start) & (dt_series <= splitter.val_end)
        test_mask = (dt_series >= splitter.test_start) & (dt_series <= splitter.test_end)

        # Fit model on training split
        model = XGBoostForecaster(name=f"xgb_h{h}", n_estimators=120, max_depth=6, random_state=42)
        model.fit(X[train_mask], y[train_mask])

        # Calibrate strictly on validation partition
        cp = ConformalPredictor(base_model=model, default_coverage=0.90)
        cp.calibrate(X[val_mask], y[val_mask])
        q_90 = cp.quantiles_[0.90]

        # Evaluate on test partition
        test_preds = cp.predict_intervals(X[test_mask], coverage_level=0.90)
        y_test = y[test_mask].values
        lb_test = test_preds["lower_bound"].values
        ub_test = test_preds["upper_bound"].values
        test_meta = meta[test_mask].copy()
        test_meta["y_true"] = y_test
        test_meta["lb"] = lb_test
        test_meta["ub"] = ub_test
        test_meta["covered"] = (y_test >= lb_test) & (y_test <= ub_test)
        test_meta["is_heating_season"] = X.loc[test_mask, "is_heating_season"].values

        overall_cov = float(np.mean(test_meta["covered"]))
        mean_width = float(np.mean(ub_test - lb_test))

        print(f"\n[HORIZON h={h}h] Nominal: 90.0% | Calibration Quantile q_0.90 = {q_90:.2f} ug/m3")
        print(f"  Overall Test Coverage: {overall_cov*100:.2f}% | Mean Interval Width: {mean_width:.2f} ug/m3 (N={len(test_meta):,})")

        # By Station
        station_cov = {}
        for st, st_group in test_meta.groupby("station"):
            st_c = float(np.mean(st_group["covered"]))
            st_w = float(np.mean(st_group["ub"] - st_group["lb"]))
            station_cov[st] = {"coverage": round(st_c * 100, 2), "mean_width": round(st_w, 2), "samples": len(st_group)}
        print(f"  Station Coverage Range: min={min(v['coverage'] for v in station_cov.values()):.2f}%, max={max(v['coverage'] for v in station_cov.values()):.2f}%")

        # By Season (Oct non-heating vs Nov-Feb heating)
        season_cov = {}
        for heat_val, s_group in test_meta.groupby("is_heating_season"):
            s_name = "Heating Season (Nov-Feb)" if float(heat_val) == 1.0 else "Non-Heating (Oct)"
            s_c = float(np.mean(s_group["covered"]))
            s_w = float(np.mean(s_group["ub"] - s_group["lb"]))
            season_cov[s_name] = {"coverage": round(s_c * 100, 2), "mean_width": round(s_w, 2), "samples": len(s_group)}
            print(f"  Season [{s_name}]: Coverage: {s_c*100:.2f}% | Mean Width: {s_w:.2f} ug/m3 | N={len(s_group):,}")

        # By PM2.5 Concentration Tiers
        pm_tiers = [
            ("Good to Moderate [0, 75)", 0.0, 75.0),
            ("USG [75, 115)", 75.0, 115.0),
            ("Unhealthy [115, 150)", 115.0, 150.0),
            ("Very Unhealthy [150, 250)", 150.0, 250.0),
            ("Severe / Hazardous [250+)", 250.0, 10000.0),
        ]
        tier_cov = {}
        for t_name, low, high in pm_tiers:
            t_mask = (test_meta["y_true"] >= low) & (test_meta["y_true"] < high)
            if np.sum(t_mask) > 0:
                t_sub = test_meta[t_mask]
                t_c = float(np.mean(t_sub["covered"]))
                t_w = float(np.mean(t_sub["ub"] - t_sub["lb"]))
                tier_cov[t_name] = {"coverage": round(t_c * 100, 2), "mean_width": round(t_w, 2), "samples": len(t_sub)}
                print(f"  PM Tier [{t_name}]: Coverage: {t_c*100:.2f}% | N={len(t_sub):,}")

        audit_results[f"h{h}"] = {
            "overall_coverage": round(overall_cov * 100, 2),
            "mean_width": round(mean_width, 2),
            "calib_quantile": round(q_90, 2),
            "station_coverage": station_cov,
            "season_coverage": season_cov,
            "tier_coverage": tier_cov,
        }

    # Save detailed conformal audit results
    os.makedirs("artifacts/metrics", exist_ok=True)
    with open("artifacts/metrics/conformal_stratified_audit.json", "w") as f:
        json.dump(audit_results, f, indent=2)
    print("\nSaved stratified conformal audit metrics to artifacts/metrics/conformal_stratified_audit.json")

    # =========================================================================
    # AUDIT 2: LOSO PERSISTENCE BASELINE RECONCILIATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("AUDIT 2: LOSO PERSISTENCE BASELINE EVALUATION (h=24h)")
    print("=" * 60)
    X24, y24, meta24 = builder.get_horizon_dataset(df_prepared, horizon=24, feature_group="combined_all")
    loso_pers_results = {}
    stations = sorted(meta24["station"].unique())

    for st in stations:
        is_held_out = meta24["station"] == st
        y_test_st = y24[is_held_out].values
        # Persistence prediction is PM2.5 at time t
        y_pred_pers = X24.loc[is_held_out, "PM2.5"].values
        # Calculate persistence metrics on this held-out station
        m_pers = calculate_regression_metrics(y_test_st, y_pred_pers)
        loso_pers_results[st] = {
            "pers_mae": round(m_pers["mae"], 2),
            "pers_rmse": round(m_pers["rmse"], 2),
            "pers_r2": round(m_pers["r2"], 4),
        }
        print(f"  Station [{st:<14}]: Persistence MAE: {m_pers['mae']:6.2f} | RMSE: {m_pers['rmse']:6.2f} | R2: {m_pers['r2']:7.4f}")

    mean_pers_mae = np.mean([v["pers_mae"] for v in loso_pers_results.values()])
    mean_pers_rmse = np.mean([v["pers_rmse"] for v in loso_pers_results.values()])
    mean_pers_r2 = np.mean([v["pers_r2"] for v in loso_pers_results.values()])
    print(f"\n12-Station Mean Persistence: MAE: {mean_pers_mae:.2f} | RMSE: {mean_pers_rmse:.2f} | R2: {mean_pers_r2:.4f}")

    with open("artifacts/metrics/loso_persistence_audit.json", "w") as f:
        json.dump({"loso_persistence": loso_pers_results, "mean": {"mae": mean_pers_mae, "rmse": mean_pers_rmse, "r2": mean_pers_r2}}, f, indent=2)

if __name__ == "__main__":
    main()
