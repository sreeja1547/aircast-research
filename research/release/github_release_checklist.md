# AirCast Research — GitHub Release & Portfolio Checklist

**Document Purpose:** Pre-Publication Verification Gate for Open-Source Portfolio Release  
**Target Repository:** AirCast Research: A Research-Driven Framework for Explainable, Uncertainty-Aware and Generalizable Multi-Horizon Air Quality Forecasting  
**Date:** November 2026  
**Auditor:** Senior ML Systems Architect & Research Release Auditor  

---

## 1. Repository Hygiene & Cleanup Checklist

- [x] **Unrelated Non-Project Files Removed:**
  - Removed stock telemetry files (`data.csv`, `test.csv`).
  - Removed extraneous photo (`2A2478DC-8517-4490-9FA7-36F9A7A542BE.JPG`).
- [x] **Duplicate Dataset Archives Removed:**
  - Removed `PRSA2017_Data_20130301-20170228.zip` after confirming 100% byte-for-byte SHA-256 identity with the uncompressed 12-station CSV files in `PRSA_Data_20130301-20170228/`.
- [x] **Legacy Inspection Scripts Retired:**
  - Removed root `audit_dataset.py` and `audit_report.json` (superseded by package pipeline and `eda_statistical_summary.json`).
- [x] **External Reversible Backup Verified:**
  - All 6 removed files backed up to `scratch/cleanup_backup/` with documented SHA-256 checksums and byte sizes.
- [x] **Ephemeral Caches Cleaned:**
  - Purged 12 `__pycache__` directories.
  - Purged `.pytest_cache/` test runner directories.
- [x] **Git Ignore Configured:**
  - Created `.gitignore` covering Python bytecode, virtual environments, editor metadata, OS artifacts, and scratch logs.
  - Whitelisted research metrics (`!artifacts/metrics/*.json`), models (`!artifacts/models/*.joblib`), plots (`!artifacts/plots/*.png`), and datasets.

---

## 2. Security, Privacy & Secrets Checklist

- [x] **Zero Hardcoded Secrets:**
  - Regex audit across all files confirmed zero API keys, passwords, bearer tokens, or cloud service credentials.
- [x] **CORS Configuration Inspected:**
  - `allow_credentials=True` in `api/main.py` verified as local research prototype configuration; CORS is open for local development.
- [x] **Zero Personal User Paths in Code:**
  - All Python files in `src/`, `api/`, `dashboard/`, `tests/`, and `scripts/` use strictly relative path resolution or `Path(__file__)`.
- [x] **Personal Username Sanitization in Documentation:**
  - Converted 52 absolute `file:///c:/Users/Dell/...` Markdown links into clean repository-relative links across all reports.
- [x] **No Uncontrolled Public Exposure:**
  - Git repository remains uninitialized. No remote GitHub repository has been created or linked without user consent.

---

## 3. Packaging, Environment & Dependency Declarations

- [x] **Tested Python Runtime Documented:**
  - Formally documented tested runtime as **Python 3.13.5 on Windows 11 Home x64**.
  - Acknowledged that broader Python 3.10+ compatibility is supported by standard syntax but specific package builds were validated on 3.13.5.
- [x] **Requirements Specification Created:**
  - Created `requirements.txt` with tested dependency bounds (`numpy`, `pandas`, `scipy`, `pyarrow`, `joblib`, `scikit-learn`, `xgboost`, `torch`, `fastapi`, `uvicorn`, `pydantic`, `streamlit`, `matplotlib`, `seaborn`, `pytest`).
- [x] **Unused & Problematic Packages Excluded:**
  - Verified that external `shap` package is **not** required because `src/explainability/shap_engine.py` calls XGBoost's native C++ TreeSHAP implementation (`pred_contribs=True`). Omitted from `requirements.txt` to prevent Windows C++ build failures.
- [x] **CPU-Only Deep Learning Declared:**
  - PyTorch CPU-only runtime verified (`CUDA_AVAILABLE = False`). Documented clearly in environment reports.

---

## 4. Model Artifact & Conformal Calibration Preservation

- [x] **Recommended Multi-Horizon Pareto Models Intact:**
  - `artifacts/models/xgboost_h1.joblib` (1.1577 MB, 180 trees, test MAE 11.57 $\mu\text{g}/\text{m}^3$, RMSE 23.33 $\mu\text{g}/\text{m}^3$, $R^2 = 0.9499$).
  - `artifacts/models/xgboost_pruned_h6.joblib` (0.4328 MB, 90 trees, test MAE 37.04 $\mu\text{g}/\text{m}^3$, RMSE 62.63 $\mu\text{g}/\text{m}^3$, $R^2 = 0.6389$).
  - `artifacts/models/xgboost_fast_h24.joblib` (0.1352 MB, 50 trees, test MAE 66.07 $\mu\text{g}/\text{m}^3$, RMSE 92.57 $\mu\text{g}/\text{m}^3$, $R^2 = 0.2121$).
  - Total Pareto deployment suite footprint: **$1.7257\text{ MB}$**.
- [x] **Normalized Conformal Calibrators Intact:**
  - `artifacts/models/conformal_calibrator_h1.joblib` (1.1743 MB).
  - `artifacts/models/conformal_calibrator_h6.joblib` (1.2534 MB).
  - `artifacts/models/conformal_calibrator_h24.joblib` (0.8918 MB).
- [x] **Baseline Research Binaries Preserved:**
  - All 26 unpruned baseline model files (Random Forest, HistGBDT, LSTM, Ridge, Linear Regression, Persistence, Standard Conformal) preserved in `artifacts/models/` for full academic reproducibility.

---

## 5. Data Pipeline & Reproducibility Checklist

- [x] **Ground-Truth Raw Data Retained:**
  - All 12 station CSV files (420,768 records) preserved in `PRSA_Data_20130301-20170228/`.
- [x] **Processed Dataset Cache Retained:**
  - `data/processed/forecasting_features_cache.parquet` (30.52 MB, 420,768 rows $\times$ 70 columns) retained to allow instantaneous testing and serving without requiring 28-second feature generation.
- [x] **Pipeline Regeneration Script Verified:**
  - `python scripts/prepare_processed_data.py` confirmed operational and reproducible from raw station files.
- [x] **Pareto Model Serialization Script Verified:**
  - `python scripts/save_and_verify_pareto_models.py` confirmed operational.

---

## 6. Serving Layer & Prototype Operational Verification

- [x] **FastAPI Application Verified:**
  - `api/main.py` initialized and route registry verified:
    - `GET /health` (Status & version)
    - `GET /model-info` (Model catalog & parameter metadata)
    - `GET /features` (Feature registry groups)
    - `GET /metrics` (Benchmark metrics summary)
    - `POST /predict` (Single-sample forecasting with conformal bounds)
    - `POST /predict/batch` (Batch vectorized forecasting)
    - `GET /docs` (Swagger UI)
- [x] **Inference Service Serving Verified:**
  - `AirCastInferenceService` verified across $+1\text{h}$, $+6\text{h}$, and $+24\text{h}$ with calibrated normalized intervals.
- [x] **Streamlit Dashboard Verified:**
  - `dashboard/app.py` imported cleanly with all 9 research tabs and utility helpers intact.
- [x] **Performance Disclosures Maintained:**
  - Honestly documented that single-record REST API P95 latency is $43.3\text{–}44.1\text{ ms}$ (unmet for $\le 15.0\text{ ms}$ SLA), while raw C++ NumPy buffer latency is $4.33\text{ ms}$.

---

## 7. Documentation & Hyperlink Integrity

- [x] **`README.md` Relative Links Verified:**
  - All 26 markdown links point to valid, existing files in `research/`, `research/efficiency/`, and `artifacts/`.
  - Zero broken links detected.
- [x] **RMSE Discrepancy Reconciliation Documented:**
  - Added `discrepancy_6_rmse_consistency_reconciliation` in `phase11_final_verified_results.json` and updated tables in `README.md` and `final_release_readiness.md`.
- [x] **LOSO Boundary Transparency:**
  - Documented that 12-fold LOSO evaluates transferability to a newly instrumented station with local telemetry ($\tau \le t$), not zero-shot estimation at uninstrumented locations.
- [x] **Conformal Uncertainty Framing:**
  - Documented empirical coverage degradation at $+24\text{h}$ lead time and severe pollution tiers ($> 250\ \mu\text{g}/\text{m}^3$) due to winter exchangeability shifts. Removed all "guaranteed coverage" claims.

---

## 8. Automated Testing & Release Gate

- [x] **Complete Pytest Suite:**
  - **52 passed, 0 failed, 6 warnings in 41.18s (100.0% pass rate).**
- [x] **GitHub File Size Limit Compliance:**
  - Largest file: `random_forest_h24.joblib` (41.86 MB).
  - 0 files exceed GitHub's 50 MB warning limit.
  - 0 files exceed GitHub's 100 MB block limit.
  - Total repository size: **185.02 MB** (well below GitHub's 1 GB target).
- [x] **Git Initialization Status:**
  - Git is **not yet initialized** in this workspace. No commits or remotes exist.
- [x] **Release Determination:**
  - **APPROVED FOR OPEN-SOURCE GITHUB PORTFOLIO RELEASE.**

---

> [!NOTE]
> **NEXT STEPS (UPON EXPLICIT USER APPROVAL):**  
> 1. Initialize local git repository (`git init`).
> 2. Stage verified files (`git add .`).
> 3. Create initial clean release commit: `feat: release AirCast Research v1.0.0-research`.
> 4. Link to user-designated GitHub remote if desired.
