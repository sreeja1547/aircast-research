# AirCast Research — Final GitHub Release Preparation Report

**Role:** Senior ML Systems Engineer & GitHub Release Manager  
**Date:** November 2026  
**Repository Working Directory:** `beijing+multi+site+air+quality+data/`  
**Target Release Branch:** `main`  
**Release Tag / Version:** `v1.0.0`  
**License:** MIT License (`Copyright (c) 2026 Yenduri Sreeja`)  
**Deployment Profile:** Production-Ready Applied ML Research Framework (CPU-Optimized)  

---

## 1. Final Repository Structure

The repository architecture reflects a modular, professional machine learning project structure optimized for immediate inspection, installation, and reproducibility by engineering recruiters and researchers:

```text
aircast-research/
├── src/                               # Core Python scientific framework (22 modules)
│   ├── data/                          # Causal imputation, alignment, loading
│   ├── features/                      # Atmospheric physics, wind, rolling transforms
│   ├── models/                        # Baseline, tree ensemble, and PyTorch architectures
│   ├── uncertainty/                   # Normalized split conformal prediction & calibrators
│   ├── explainability/                # Native C++ TreeSHAP attribution engine
│   ├── experiments/                   # Benchmarking, ablation, LOSO execution logic
│   └── evaluation/                    # Metrics, 12-fold LOSO, sensor robustness suite
├── api/                               # FastAPI REST serving backend (4 modules)
│   ├── main.py                        # Endpoint routing, lifespan events, middleware
│   ├── schemas.py                     # Strict Pydantic v2 request/response schemas
│   └── services.py                    # Multi-horizon inference & conformal prediction service
├── dashboard/                         # Streamlit interactive research application
│   └── app.py                         # 9-panel comprehensive research GUI
├── scripts/                           # Reproducible automation & audit pipeline (19 scripts)
│   ├── prepare_processed_data.py      # Causal feature derivation (~28 sec)
│   ├── save_and_verify_pareto_models.py # Model verification & scoring
│   ├── generate_shap_plots.py         # Exact TreeSHAP waterfall generation
│   └── run_all_experiments.py         # End-to-end experiment replication
├── tests/                             # Automated pytest regression suite (9 test files)
│   ├── test_api.py                    # REST API route & inference payload testing
│   ├── test_conformal.py              # Split conformal calibration & quantile tests
│   ├── test_efficiency.py             # Memory downcasting & latency budget tests
│   ├── test_evaluation.py             # Statistical metric & masking tests
│   ├── test_explainability.py         # C++ TreeSHAP attribution tests
│   ├── test_feature_engineering.py    # Wind vectors & rolling window causality tests
│   ├── test_models.py                 # Multi-paradigm model contract tests
│   ├── test_preprocessing.py          # StationAwareCausalImputer tests
│   └── test_robustness.py             # Sensor noise & telemetry dropout tests
├── artifacts/                         # Research evidence, models, and figures
│   ├── models/                        # 6 recommended Pareto models & conformal calibrators
│   ├── metrics/                       # 26 verified evaluation JSON summaries
│   └── plots/                         # 21 high-resolution publication figures
├── research/                          # Comprehensive technical reports (35 reports)
│   ├── efficiency/                    # 14 Phase 11 computational optimization reports
│   └── release/                       # 7 release audits, checklists, & governance reports
├── data/                              # Processed data cache
│   └── processed/
│       └── forecasting_features_cache.parquet  # 420,768 rows x 70 cols (30.52 MB)
├── README.md                          # 18-section professional project documentation
├── requirements.txt                   # Bounded, verified dependencies (Python 3.13.5)
├── LICENSE                            # MIT License (Copyright (c) 2026 Yenduri Sreeja)
└── .gitignore                         # Strict exclusion matrix for caches, datasets, & secrets
```

---

## 2. Committed Files & Repository Footprint Audit

- **Total Committed Files in Git:** **160 files**
- **Total Tracked Git Repository Size:** **$45.92\text{ MB}$**
- **Size Reduction from Unoptimized Baseline ($154.08\text{ MB} \to 45.92\text{ MB}$):** **$-70.20\%$ footprint reduction**
- **GitHub Hard Limit Check ($100\text{ MB}$):** **0 files exceed 100 MB (100% compliant)**
- **GitHub Warning Threshold Check ($50\text{ MB}$):** **0 files exceed 50 MB (100% compliant)**
- **Largest Tracked File:** `data/processed/forecasting_features_cache.parquet` ($30.52\text{ MB}$).
- **Largest Model File:** `artifacts/models/conformal_calibrator_h6.joblib` ($1.25\text{ MB}$).

---

## 3. Retained Models & Conformal Calibrators

The Git repository tracks the complete, production-ready **Pareto Deployment Suite** required for end-to-end multi-horizon serving, API endpoints, and dashboard simulation:

| Lead Horizon | Model Artifact Filename | Purpose / Architectural Profile | Disk Size | Test MAE | Test RMSE | Test $R^2$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **$+1\text{h}$** | `artifacts/models/xgboost_h1.joblib` | Immediate Inertial Forecaster (180 trees, max depth 7) | $1.158\text{ MB}$ | **11.5723** | **23.3344** | **0.9499** |
| **$+6\text{h}$** | `artifacts/models/xgboost_pruned_h6.joblib` | Mesoscale Boundary Layer Forecaster (90 trees, max depth 6) | $0.433\text{ MB}$ | **37.0350** | **62.6263** | **0.6389** |
| **$+24\text{h}$** | `artifacts/models/xgboost_fast_h24.joblib` | Synoptic Transport Forecaster (50 trees, max depth 5) | $0.136\text{ MB}$ | **66.0738** | **92.5722** | **0.2121** |
| **$+1\text{h}$** | `artifacts/models/conformal_calibrator_h1.joblib` | Normalized Adaptive Split Conformal Calibrator ($1\text{h}$) | $1.174\text{ MB}$ | — | — | — |
| **$+6\text{h}$** | `artifacts/models/conformal_calibrator_h6.joblib` | Normalized Adaptive Split Conformal Calibrator ($6\text{h}$) | $1.253\text{ MB}$ | — | — | — |
| **$+24\text{h}$** | `artifacts/models/conformal_calibrator_h24.joblib` | Normalized Adaptive Split Conformal Calibrator ($24\text{h}$) | $0.892\text{ MB}$ | — | — | — |

**Total Storage for Recommended Deployment Suite:** **$5.046\text{ MB}$**.

---

## 4. Excluded Artifacts & Technical Justification

To adhere to open-source repository best practices and eliminate unnecessary repository bloat, **23 experimental baseline model artifacts ($108.17\text{ MB}$ total)** were excluded from Git tracking via `.gitignore`. 

> **Important Preservation Guarantee:** Every excluded model remains **100% intact locally on disk in `artifacts/models/`**. Their evaluation metrics, comparisons, and convergence curves are permanently preserved in `artifacts/metrics/` and technical documentation.

| Excluded Artifact Group | Filenames | Combined Size | Exclusion Justification | Local Disk Status |
| :--- | :--- | :---: | :--- | :---: |
| **Unpruned Random Forests** | `random_forest_h{1,6,24}.joblib` | **$100.67\text{ MB}$** | Experimental baselines from Phase 4/11. Bulky ($41.86\text{ MB}$ max). Completely superseded by the Pareto XGBoost suite. Not used by API or dashboard. | **Preserved Locally** |
| **Linear & Ridge Baselines** | `linear_regression_h{1,6,24}.joblib`, `ridge_h{1,6,24}.joblib` | $0.018\text{ MB}$ | Parametric baselines from Phase 4 benchmarking. Full metrics captured in `artifacts/metrics/experiment1_benchmarking.json`. | **Preserved Locally** |
| **Persistence Baselines** | `persistence_h{1,6,24}.joblib` | $0.003\text{ MB}$ | Naive particulate inertia baseline. Mathematical identity requiring zero weights. | **Preserved Locally** |
| **Histogram GBDT Baselines** | `hist_gradient_boosting_h{1,6,24}.joblib` | $1.743\text{ MB}$ | Intermediate comparative tree baseline. Subsumed by tuned XGBoost forecasters. | **Preserved Locally** |
| **Deep Learning Baselines** | `lstm_h{1,6,24}.joblib` | $0.754\text{ MB}$ | Recurrent sequence baselines from Phase 5. Metrics documented in `deep_learning_report.md`. | **Preserved Locally** |
| **Unpruned XGBoost Models** | `xgboost_h6.joblib`, `xgboost_h24.joblib` | $1.651\text{ MB}$ | Baseline 180-tree models prior to Phase 11 tree pruning. Superseded by pruned Pareto models. | **Preserved Locally** |
| **Static Conformal Calibrators**| `conformal_calibrator_std_h{1,6,24}.joblib` | $3.319\text{ MB}$ | Early homoscedastic static margin calibrators from Phase 7. Superseded by normalized adaptive calibrators. | **Preserved Locally** |

---

## 5. Local Raw Dataset Preservation & Detection Verification

- **Local File Preservation:** All 12 raw station telemetry CSV files ($31.00\text{ MB}$) are **100% preserved locally on disk** in `PRSA_Data_20130301-20170228/`:
  - `PRSA_Data_Aotizhongxin_20130301-20170228.csv`
  - `PRSA_Data_Changping_20130301-20170228.csv`
  - `PRSA_Data_Dingling_20130301-20170228.csv`
  - `PRSA_Data_Dongsi_20130301-20170228.csv`
  - `PRSA_Data_Guanyuan_20130301-20170228.csv`
  - `PRSA_Data_Gucheng_20130301-20170228.csv`
  - `PRSA_Data_Huairou_20130301-20170228.csv`
  - `PRSA_Data_Nongzhanguan_20130301-20170228.csv`
  - `PRSA_Data_Shunyi_20130301-20170228.csv`
  - `PRSA_Data_Tiantan_20130301-20170228.csv`
  - `PRSA_Data_Wanliu_20130301-20170228.csv`
  - `PRSA_Data_Wanshouxigong_20130301-20170228.csv`
- **Git Exclusion:** Verified via `git status --ignored -s` that `PRSA_Data_20130301-20170228/` is properly ignored and will never be pushed to GitHub.
- **Automated Ingestion Verification:** `src/data/loader.py` and `scripts/prepare_processed_data.py` inspect `DEFAULT_RAW_DIR = "PRSA_Data_20130301-20170228"`, ensuring that any user who extracts the official UCI dataset will have their data automatically ingested.

---

## 6. Functional Application Verification

### 6.1 Inference Service (`AirCastInferenceService`)
Direct programmatic scoring verified across all planning horizons using production artifacts:
- **$+1\text{h}$ Point Forecast:** $63.40\ \mu\text{g}/\text{m}^3$ | Conformal Bounds: $[46.04, 80.75]\ \mu\text{g}/\text{m}^3$ | Latency: $97.80\text{ ms}$ (Cold start)
- **$+6\text{h}$ Point Forecast:** $56.66\ \mu\text{g}/\text{m}^3$ | Conformal Bounds: $[14.44, 98.88]\ \mu\text{g}/\text{m}^3$ | Latency: $78.58\text{ ms}$
- **$+24\text{h}$ Point Forecast:** $67.04\ \mu\text{g}/\text{m}^3$ | Conformal Bounds: $[2.75, 131.34]\ \mu\text{g}/\text{m}^3$ | Latency: $67.34\text{ ms}$

### 6.2 REST API Backend (`api/main.py`)
- Verified all 10 registered routes (`/health`, `/model-info`, `/features`, `/metrics`, `/predict`, `/predict/batch`, `/docs`, `/redoc`, `/openapi.json`).
- Schema validation active with Pydantic v2.

### 6.3 Research Streamlit Dashboard (`dashboard/app.py`)
- Imported cleanly without runtime syntax or logic errors.
- Visualizes 9 interactive scientific panels (EDA distributions, model comparisons, feature ablations, seasonal shifts, 12-fold LOSO generalisation, conformal calibration curves, TreeSHAP waterfalls, and scenario simulator).

---

## 7. Automated Pytest Regression Results

The complete regression test suite was executed in the test environment (Python 3.13.5 on Windows 11 x64):

```text
pytest tests/ -q
.................................................... [100%]
======================= 52 passed, 6 warnings in 37.82s =======================
```

**Pass Rate:** **52 / 52 Passed (100.0%)**  
- `tests/test_api.py`: 5 passed
- `tests/test_conformal.py`: 5 passed
- `tests/test_efficiency.py`: 6 passed
- `tests/test_evaluation.py`: 6 passed
- `tests/test_explainability.py`: 4 passed
- `tests/test_feature_engineering.py`: 8 passed
- `tests/test_models.py`: 8 passed
- `tests/test_preprocessing.py`: 5 passed
- `tests/test_robustness.py`: 5 passed

---

## 8. Final Git Release Commit

```text
[main 9c8ad82] Initial release: AirCast Research v1.0.0
 160 files changed, 32742 insertions(+)
```

- **Commit Hash:** `9c8ad82` (Amended root commit)
- **Branch:** `main`
- **Author:** `sreeja1547 <sreejayenduri1547@gmail.com>`
- **Committed Files:** **160**
- **Working Tree:** Clean (`nothing to commit, working tree clean`)

---

## 9. Remaining Warnings & Disclosures

1. **Hardware Execution Disclosure:** All benchmarks and inference pipelines operate strictly on commodity **CPU hardware**. No GPU acceleration is utilized or required.
2. **Local Prototype Notice:** Serving components are engineered as transparent research prototypes for interactive exploration and scientific audit. They are not hardened multi-tenant cloud services.
3. **Distribution Shift Boundary:** Conformal prediction intervals achieve verified empirical coverage on typical conditions but contract during extreme winter heating stagnation episodes ($> 250\ \mu\text{g}/\text{m}^3$). No theoretical $90\%$ coverage guarantee is claimed under non-stationary seasonal shifts.
4. **Third-Party Dataset:** The PRSA 2017 dataset is authored by Song Chen et al. (Tsinghua / Peking University). Full attribution is provided in `README.md` and `LICENSE` disclosures.

---

## 10. Final Release Gate Recommendation

# **FINAL STATUS: 100% READY FOR PUBLIC GITHUB RELEASE**

The repository is clean, fully verified, free of secrets or bloat, and packaged for presentation to hiring managers, research reviewers, and applied ML teams.

### Deployment Instructions:
```bash
git remote add origin https://github.com/sreeja1547/<your-repo-name>.git
git push -u origin main
```
