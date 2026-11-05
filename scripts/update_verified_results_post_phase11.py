"""Update phase11_final_verified_results.json with rigorous Post-Phase 11 audit measurements."""

import json
import datetime
from pathlib import Path

def main():
    p = Path("artifacts/metrics/phase11_final_verified_results.json")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["audit_version"] = "2.0.0-final-release"
    data["timestamp"] = datetime.datetime.now().isoformat()

    # 1. Conformal Coverage Audit
    data["conformal_coverage_audit"] = {
        "nominal_confidence_target": 0.90,
        "pooled_empirical_coverage": 84.72,
        "total_evaluated_test_samples": 127326,
        "historical_91_4_percent_source": (
            "The 91.4% figure originated in Phase 7 (research/uncertainty_and_explainability.md) "
            "from a single-horizon static split conformal predictor with a uniform fixed margin "
            "of 56.20 ug/m3 evaluated on 42,278 test samples (38,642 covered). It did not represent "
            "horizon-specific normalized conformal coverage across all lead times."
        ),
        "horizon_breakdown": {
            "h1": {
                "calibrator_artifact": "conformal_calibrator_h1.joblib",
                "method": "normalized_adaptive_conformal",
                "overall_empirical_coverage": 89.07,
                "coverage_gap_pct": -0.93,
                "mean_interval_width": 44.88,
                "median_interval_width": 35.54,
                "test_samples": 42554,
                "severe_pollution_tier_coverage": {
                    "good_0_to_35": {"coverage": 92.53, "mean_width": 20.82, "samples": 15818},
                    "moderate_35_to_75": {"coverage": 85.80, "mean_width": 33.17, "samples": 7322},
                    "unhealthy_sensitive_75_to_115": {"coverage": 87.29, "mean_width": 44.10, "samples": 5436},
                    "unhealthy_115_to_150": {"coverage": 86.83, "mean_width": 54.72, "samples": 3584},
                    "very_unhealthy_150_to_250": {"coverage": 88.61, "mean_width": 72.18, "samples": 6718},
                    "hazardous_severe_above_250": {"coverage": 86.34, "mean_width": 113.39, "samples": 3676}
                }
            },
            "h6": {
                "calibrator_artifact": "conformal_calibrator_h6.joblib",
                "method": "normalized_adaptive_conformal",
                "overall_empirical_coverage": 85.26,
                "coverage_gap_pct": -4.74,
                "mean_interval_width": 131.29,
                "median_interval_width": 111.77,
                "test_samples": 42494,
                "severe_pollution_tier_coverage": {
                    "good_0_to_35": {"coverage": 82.93, "mean_width": 60.71, "samples": 15818},
                    "moderate_35_to_75": {"coverage": 89.25, "mean_width": 100.58, "samples": 7322},
                    "unhealthy_sensitive_75_to_115": {"coverage": 91.81, "mean_width": 131.34, "samples": 5436},
                    "unhealthy_115_to_150": {"coverage": 88.91, "mean_width": 163.03, "samples": 3580},
                    "very_unhealthy_150_to_250": {"coverage": 87.87, "mean_width": 219.99, "samples": 6662},
                    "hazardous_severe_above_250": {"coverage": 69.40, "mean_width": 304.46, "samples": 3676}
                }
            },
            "h24": {
                "calibrator_artifact": "conformal_calibrator_h24.joblib",
                "method": "normalized_adaptive_conformal",
                "overall_empirical_coverage": 79.81,
                "coverage_gap_pct": -10.19,
                "mean_interval_width": 181.15,
                "median_interval_width": 174.62,
                "test_samples": 42278,
                "severe_pollution_tier_coverage": {
                    "good_0_to_35": {"coverage": 79.97, "mean_width": 158.96, "samples": 15818},
                    "moderate_35_to_75": {"coverage": 98.98, "mean_width": 147.14, "samples": 7322},
                    "unhealthy_sensitive_75_to_115": {"coverage": 92.82, "mean_width": 165.57, "samples": 5435},
                    "unhealthy_115_to_150": {"coverage": 87.48, "mean_width": 193.29, "samples": 3554},
                    "very_unhealthy_150_to_250": {"coverage": 71.95, "mean_width": 228.81, "samples": 6477},
                    "hazardous_severe_above_250": {"coverage": 28.05, "mean_width": 271.82, "samples": 3672}
                }
            }
        },
        "scientific_interpretation": (
            "Conformal coverage under non-stationary atmospheric time series degrades from 89.1% at +1h to 79.8% at +24h. "
            "Under finite-sample non-exchangeability caused by seasonal winter inversion regimes, empirical coverage on "
            "extreme events (>250 ug/m3) drops to 28.05% at 24 hours. The framework does not guarantee 90% coverage on unseen regimes."
        )
    }

    # 2. API Latency Reconciliation
    data["api_latency_reconciliation"] = {
        "budget_target_ms": 15.0,
        "budget_status_summary": "UNMET for end-to-end REST API single-record serving (P95 is 43-44 ms); MET for direct C++ NumPy array inference (4.33 ms).",
        "horizons": {
            "h1": {
                "model": "xgboost_h1.joblib (180 trees, depth 7)",
                "cold_warmup_latency_ms": 73.91,
                "steady_state_mean_ms": 28.05,
                "steady_state_median_ms": 35.98,
                "steady_state_p95_ms": 44.13,
                "steady_state_p99_ms": 57.86,
                "breakdown": {
                    "feature_prep_ms": 3.54,
                    "model_inference_ms": 24.49,
                    "conformal_calculation_ms": 0.02
                },
                "meets_15ms_budget": False
            },
            "h6": {
                "model": "xgboost_pruned_h6.joblib (90 trees, depth 6)",
                "cold_warmup_latency_ms": 149.20,
                "steady_state_mean_ms": 35.04,
                "steady_state_median_ms": 38.00,
                "steady_state_p95_ms": 43.36,
                "steady_state_p99_ms": 66.70,
                "breakdown": {
                    "feature_prep_ms": 4.45,
                    "model_inference_ms": 30.56,
                    "conformal_calculation_ms": 0.02
                },
                "meets_15ms_budget": False
            },
            "h24": {
                "model": "xgboost_fast_h24.joblib (50 trees, depth 5)",
                "cold_warmup_latency_ms": 143.29,
                "steady_state_mean_ms": 24.72,
                "steady_state_median_ms": 13.84,
                "steady_state_p95_ms": 43.32,
                "steady_state_p99_ms": 48.43,
                "breakdown": {
                    "feature_prep_ms": 3.16,
                    "model_inference_ms": 21.54,
                    "conformal_calculation_ms": 0.02
                },
                "meets_15ms_budget": False
            }
        },
        "source_of_delay": (
            "84-87% of single-record API latency stems from XGBoost C++ kernel inference on a single 1-row Pandas DataFrame "
            "(21-30 ms), while feature dictionary parsing accounts for 12-16% (3-4.5 ms). Conformal margin arithmetic takes < 0.02 ms. "
            "Cold warmup calls take 74-149 ms due to lazy model unpickling and OpenMP thread pool allocation."
        )
    }

    # 3. Numerical Equivalence Verification
    data["numerical_equivalence_audit"] = {
        "same_model_buffer_vs_pandas": {
            "h1": {"max_diff": 0.0, "mean_diff": 0.0, "numerically_identical": True},
            "h6": {"max_diff": 0.0, "mean_diff": 0.0, "numerically_identical": True},
            "h24": {"max_diff": 0.0, "mean_diff": 0.0, "numerically_identical": True},
            "conclusion": "Vectorized buffer inference on the SAME model artifact produces exact 0.0 bit-level numerical equivalence."
        },
        "cross_model_comparison_baseline_vs_recommended": {
            "h1": {
                "baseline_artifact": "xgboost_h1.joblib (180 trees)",
                "optimized_artifact": "xgboost_h1.joblib (180 trees)",
                "max_diff": 0.0,
                "mean_diff": 0.0,
                "delta_mae": 0.0,
                "is_structurally_identical": True
            },
            "h6": {
                "baseline_artifact": "xgboost_h6.joblib (180 trees, depth 7)",
                "optimized_artifact": "xgboost_pruned_h6.joblib (90 trees, depth 6)",
                "max_diff": 89.4592,
                "mean_diff": 4.9380,
                "baseline_mae": 36.9541,
                "optimized_mae": 37.0350,
                "delta_mae": 0.0809,
                "is_structurally_identical": False
            },
            "h24": {
                "baseline_artifact": "xgboost_h24.joblib (180 trees, depth 7)",
                "optimized_artifact": "xgboost_fast_h24.joblib (50 trees, depth 5)",
                "max_diff": 103.4706,
                "mean_diff": 9.1131,
                "baseline_mae": 67.2555,
                "optimized_mae": 66.0738,
                "delta_mae": -1.1817,
                "is_structurally_identical": False
            },
            "conclusion": "Recommended optimized models for +6h and +24h are pruned architectural variants, NOT numerically identical to baselines."
        }
    }

    # 4. Spatial Generalization (12-Fold LOSO)
    data["spatial_generalization_audit"] = {
        "experiment": "Experiment 7: 12-Fold Leave-One-Station-Out",
        "architecture": "XGBoost (Histogram Gradient Boosting)",
        "horizon_hours": 24,
        "n_stations": 12,
        "macro_averaged_mae": 46.33,
        "macro_averaged_mae_std": 1.65,
        "macro_averaged_rmse": 64.48,
        "macro_averaged_r2": 0.3550,
        "total_evaluated_samples": 411165,
        "methodological_caveat": (
            "Held-out stations retain access to their own local historical telemetry (autoregressive lags and weather) "
            "observed up to time t. This evaluates domain transfer to a new instrumented monitoring site, NOT prediction "
            "at unmonitored or uninstrumented geographic coordinates."
        ),
        "folds": [
            {"fold": 1, "held_out_station": "Aotizhongxin", "domain": "Urban Core", "samples": 34067, "mae": 45.85, "rmse": 63.61, "r2": 0.4009},
            {"fold": 2, "held_out_station": "Changping", "domain": "Suburban", "samples": 34218, "mae": 43.83, "rmse": 59.90, "r2": 0.3148},
            {"fold": 3, "held_out_station": "Dingling", "domain": "Regional Background", "samples": 34213, "mae": 44.15, "rmse": 60.08, "r2": 0.3098},
            {"fold": 4, "held_out_station": "Dongsi", "domain": "Urban Core", "samples": 34242, "mae": 48.03, "rmse": 67.32, "r2": 0.3960},
            {"fold": 5, "held_out_station": "Guanyuan", "domain": "Urban Core", "samples": 34376, "mae": 46.24, "rmse": 63.80, "r2": 0.3791},
            {"fold": 6, "held_out_station": "Gucheng", "domain": "Industrial Corridor", "samples": 34346, "mae": 46.97, "rmse": 67.24, "r2": 0.3413},
            {"fold": 7, "held_out_station": "Huairou", "domain": "Suburban", "samples": 34039, "mae": 43.45, "rmse": 59.30, "r2": 0.3075},
            {"fold": 8, "held_out_station": "Nongzhanguan", "domain": "Urban Core", "samples": 34364, "mae": 47.85, "rmse": 67.10, "r2": 0.3951},
            {"fold": 9, "held_out_station": "Shunyi", "domain": "Suburban", "samples": 34079, "mae": 47.45, "rmse": 66.56, "r2": 0.3293},
            {"fold": 10, "held_out_station": "Tiantan", "domain": "Urban Core", "samples": 34315, "mae": 46.27, "rmse": 64.26, "r2": 0.3701},
            {"fold": 11, "held_out_station": "Wanliu", "domain": "Urban Core", "samples": 34610, "mae": 47.10, "rmse": 65.34, "r2": 0.3642},
            {"fold": 12, "held_out_station": "Wanshouxigong", "domain": "Urban Core", "samples": 34296, "mae": 48.72, "rmse": 69.24, "r2": 0.3522}
        ]
    }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {p} ({p.stat().st_size:,} bytes) successfully!")

if __name__ == "__main__":
    main()
