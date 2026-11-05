# AirCast Research — Safe Portfolio Cleanup Plan

**Auditor Role:** Principal Applied Scientist, ML Systems Architect & Independent Release Reviewer  
**Audit Date:** November 2026  
**Status:** Audit & Classification Complete — Awaiting Approval Prior to Any File Deletion  
**Scope:** 238 Files, 193.26 MB Total Footprint across 12 Monitored Station Data Streams  

---

## 1. Executive Context & Principles

The objective of this cleanup plan is to prepare the **AirCast Research** codebase for public GitHub portfolio release without compromising scientific integrity, reproducible benchmark evidence, or application runtime functionality.

### Non-Negotiable Governance Rules
1. **Zero Scientific Regression:** No models are retrained, no architectures or hyperparameters are altered, and no ML features are added or removed.
2. **Preservation of Evidence:** Original experimental evidence (`artifacts/metrics/*.json`, `artifacts/plots/*.png`, `research/*.md`) is preserved in its entirety.
3. **Artifact Integrity:** All 3 recommended Pareto deployment models (`xgboost_h1.joblib`, `xgboost_pruned_h6.joblib`, `xgboost_fast_h24.joblib`) and their respective normalized conformal calibrators remain intact and loadable.
4. **Serving Integrity:** FastAPI backend (`api/main.py`) and Streamlit dashboard (`dashboard/app.py`) remain fully functional.
5. **No Blind Deletion:** No files are deleted during Step 1. All classifications below are proposed for explicit user review and approval.

---

## 2. File Classification Matrix

All 238 files across the repository have been audited and assigned to one of five distinct operational tiers:

```
+---------------------------------------------------------------------------------------+
|                                    REPOSITORY AUDIT                                   |
|                                238 Files (193.26 MB)                                  |
+--------------------+-------------------+--------------------+------------+------------+
| Tier 1: Essential  | Tier 2: Evidence  | Tier 3: Generated  | Tier 4:    | Tier 5:    |
| Core Pipeline      | Metrics & Reports | Caches & Zips      | Candidates | Large      |
| (Must Remain)      | (Must Remain)     | (Retain / Config)  | for Removal| Files      |
+--------------------+-------------------+--------------------+------------+------------+
| 35 Files (3.8 MB)  | 98 Files (127 MB) | 13 Files (38.1 MB) | 17 Items   | 4 Files    |
+--------------------+-------------------+--------------------+------------+------------+
```

---

### Tier 1: Essential Source Code & Core Serving Artifacts (Must Remain)

These files constitute the core scientific package, production API, interactive dashboard, automated test suite, and recommended deployment models. Deletion or corruption of any file in this tier would break application serving or test execution.

| Category | Relative Path / Directory | File Count | Size | Purpose & Justification |
| :--- | :--- | :---: | :---: | :--- |
| **Core Scientific Library** | `src/` (all 8 subpackages) | 22 files | 350 KB | Preprocessing, causal imputation, feature engineering, models, explainability, conformal prediction, and evaluation engines. |
| **Production REST API** | `api/main.py`, `api/schemas.py`, `api/services.py` | 3 files | 40 KB | FastAPI application, Pydantic schemas, and `AirCastInferenceService` memory-buffered serving layer. |
| **Research Dashboard** | `dashboard/app.py` | 1 file | 40 KB | 9-tab interactive Streamlit scientific exploration interface. |
| **Automated Test Suite** | `tests/test_*.py` | 9 files | 220 KB | 52-test automated regression suite verifying data causality, model inference, API endpoints, and conformal coverage. |
| **Recommended Models** | `artifacts/models/xgboost_h1.joblib` | 1 file | 1.16 MB | Pareto-optimal forecaster for $h=+1\text{h}$ lead time (180 trees). |
| **Recommended Models** | `artifacts/models/xgboost_pruned_h6.joblib` | 1 file | 0.43 MB | Pareto-optimal forecaster for $h=+6\text{h}$ lead time (90 trees). |
| **Recommended Models** | `artifacts/models/xgboost_fast_h24.joblib` | 1 file | 0.14 MB | Pareto-optimal forecaster for $h=+24\text{h}$ lead time (50 trees). |
| **Normalized Conformal** | `artifacts/models/conformal_calibrator_h1.joblib` | 1 file | 1.17 MB | Horizon-specific nonconformity calibrator for $+1\text{h}$. |
| **Normalized Conformal** | `artifacts/models/conformal_calibrator_h6.joblib` | 1 file | 1.25 MB | Horizon-specific nonconformity calibrator for $+6\text{h}$. |
| **Normalized Conformal** | `artifacts/models/conformal_calibrator_h24.joblib` | 1 file | 0.89 MB | Horizon-specific nonconformity calibrator for $+24\text{h}$. |
| **Core Pipeline Scripts**| `scripts/prepare_processed_data.py` | 1 file | 2.65 KB | Re-generates feature cache from raw CSV files in < 30 seconds. |
| **Core Pipeline Scripts**| `scripts/save_and_verify_pareto_models.py` | 1 file | 5.36 KB | Trains and serializes recommended Pareto suite. |
| **Documentation** | `README.md` | 1 file | 34 KB | Primary repository documentation and research index. |

---

### Tier 2: Research Evidence & Benchmarking Artifacts (Must Remain)

These files provide verifiable proof of all 11 development phases, peer review audits, ablation studies, and efficiency experiments.

| Category | Relative Path / Directory | File Count | Size | Purpose & Justification |
| :--- | :--- | :---: | :---: | :--- |
| **Metric Evidence Artifacts** | `artifacts/metrics/*.json` | 26 files | 0.43 MB | Machine-readable primary metrics for Experiments 1–8, Phase 11 stages 0–10, and final verified results. |
| **Scientific Figures** | `artifacts/plots/*.png` | 21 files | 9.03 MB | High-resolution publication plots (EDA, TreeSHAP, Pareto frontiers, LOSO generalisation, latency comparisons). |
| **Phases 1–10 Research Reports** | `research/*.md` | 16 files | 0.20 MB | Comprehensive methodology reports (EDA, feature formulation, benchmarking, generalisation, conformal calibration, audit). |
| **Phase 11 Efficiency Reports** | `research/efficiency/*.md` | 14 files | 0.11 MB | Stage-by-stage efficiency reports (hardware audit, memory optimization, tree pruning, API profiling, final reconciliation). |
| **Baseline Benchmark Models** | `artifacts/models/hist_gradient_boosting_h*.joblib` | 3 files | 1.74 MB | Stage 1 baseline models across 3 horizons. |
| **Baseline Benchmark Models** | `artifacts/models/lstm_h*.joblib` | 3 files | 0.75 MB | PyTorch LSTM sequence forecasters across 3 horizons. |
| **Baseline Benchmark Models** | `artifacts/models/ridge_h*.joblib`, `linear_regression_h*.joblib` | 6 files | 0.04 MB | OLS and Ridge linear baselines across 3 horizons. |
| **Baseline Benchmark Models** | `artifacts/models/persistence_h*.joblib` | 3 files | 0.01 MB | Particulate inertia persistence forecasters. |
| **Baseline Benchmark Models** | `artifacts/models/xgboost_h6.joblib`, `xgboost_h24.joblib` | 2 files | 1.65 MB | Unpruned baseline XGBoost models for comparison against Pareto suite. |
| **Standard Conformal Calibrators** | `artifacts/models/conformal_calibrator_std_h*.joblib` | 3 files | 3.32 MB | Standard (un-normalized) conformal calibration engines. |
| **Experimental Reproduction Scripts** | `scripts/measure_efficiency_baseline.py` to `run_regression_verification.py` | 18 files | 0.18 MB | Automated experiment runners for Phase 11 stages 1–10 and integrity audits. |

---

### Tier 3: Reproducible Generated Data & Caches (Retain / Configuration Option)

These files contain processed data or raw archive archives that can be regenerated or downloaded, but provide immediate out-of-the-box convenience for someone cloning the repository.

| Item | Path | Size | Retention Recommendation | Operational Impact If Removed |
| :--- | :--- | :---: | :--- | :--- |
| **Processed Feature Cache** | `data/processed/forecasting_features_cache.parquet` | 30.52 MB | **RETAIN** (Recommended) | If retained, users can clone the repo and immediately run `pytest`, `uvicorn`, and `streamlit` without running preprocessing. Can be regenerated in 28s via `prepare_processed_data.py`. |
| **Raw Station CSV Files** | `PRSA_Data_20130301-20170228/*.csv` (12 files) | 31.00 MB | **RETAIN** | Ground-truth historical air quality measurements for the 12 monitoring stations (March 2013 – February 2017). Necessary for raw data audits. |
| **Raw Dataset Zip Archive** | `PRSA2017_Data_20130301-20170228.zip` (in root) | 7.59 MB | **CANDIDATE FOR REMOVAL** | Redundant duplicate of the already-unzipped `PRSA_Data_20130301-20170228/` directory. |

---

### Tier 4: Temporary, Unrelated, and Cache Files (Candidates for Removal)

These files are completely safe to remove. They represent extraneous files from previous downloads, legacy Phase 1 inspection scripts, or transient Python execution caches.

| Item | Path / Location | Size | Type | Rationale for Proposed Removal |
| :--- | :--- | :---: | :--- | :--- |
| **1. Stock Market Data** | `data.csv` (root) | 37.8 KB | Extraneous file | Contains 2018 historical stock price telemetry (`Open,High,Low,Close,Volume`). Completely unrelated to Beijing atmospheric forecasting. |
| **2. Stock Market Data** | `test.csv` (root) | 35.2 KB | Extraneous file | Contains 2018 historical stock price telemetry (`Open,High,Low,Close,Volume`). Completely unrelated to air quality research. |
| **3. Leftover Image File** | `2A2478DC-8517-4490-9FA7-36F9A7A542BE.JPG` (root) | 158.8 KB | Extraneous file | Standalone JPEG from May 2023. Not referenced in any documentation, test, dashboard, or research report. |
| **4. Legacy Audit Script** | `audit_dataset.py` (root) | 8.8 KB | Legacy script | Early Phase 1 standalone inspection script. Superseded by `src/data/loader.py` and `scripts/prepare_processed_data.py`. |
| **5. Legacy Audit Report** | `audit_report.json` (root) | 23.3 KB | Legacy artifact | Early Phase 1 dataset audit JSON. Superseded by `artifacts/metrics/eda_statistical_summary.json` and `research/eda_report.md`. |
| **6. Python Bytecode Caches** | `**/__pycache__/*.pyc` (12 directories) | ~350 KB | Cache directories | Ephemeral compiled Python bytecode. Should be excluded via `.gitignore`. |
| **7. Pytest Runner Cache** | `.pytest_cache/` (6 internal files) | ~15 KB | Test cache | Ephemeral test runner cache. Should be excluded via `.gitignore`. |

---

### Tier 5: Large Files & GitHub Storage Consideration

GitHub enforces strict file and repository size limits:
- **Maximum File Size (Hard Block):** 100.0 MB.
- **Maximum Recommended File Size (Warning):** 50.0 MB.
- **Recommended Total Repository Size:** < 1.0 GB.

#### Audit of Repository's Largest Files
```
Rank  File Path                                           Size (MB)   GitHub Status (<50 MB)
--------------------------------------------------------------------------------------------
1     artifacts/models/random_forest_h24.joblib           41.86 MB    PASS (Below 50 MB threshold)
2     artifacts/models/random_forest_h6.joblib            34.22 MB    PASS (Below 50 MB threshold)
3     data/processed/forecasting_features_cache.parquet   30.52 MB    PASS (Below 50 MB threshold)
4     artifacts/models/random_forest_h1.joblib            24.59 MB    PASS (Below 50 MB threshold)
5     PRSA2017_Data_20130301-20170228.zip                 7.59 MB    PASS (Below 50 MB threshold)
6     artifacts/plots/eda_correlation_matrix.png          0.77 MB    PASS (Well below threshold)
```

#### Strategic Options for Baseline Random Forest Models ($100.67\text{ MB}$ Total)
The three unpruned baseline Random Forest models (`random_forest_h1.joblib`, `random_forest_h6.joblib`, `random_forest_h24.joblib`) together consume **$100.67\text{ MB}$** (over 52% of the repository's total storage).

- **Option A (Full Archival Preservation — Recommended for Research Rigor):**
  - Retain all 3 Random Forest baseline files in `artifacts/models/`.
  - *Feasibility:* Permitted by GitHub because each individual file is under 42 MB (below the 50 MB warning threshold). The entire repository is only **193.26 MB**, easily fitting well within GitHub's 1 GB target.
- **Option B (Pareto-Only Lean Repository):**
  - Keep only the 3 recommended Pareto models (`xgboost_h1.joblib`, `xgboost_pruned_h6.joblib`, `xgboost_fast_h24.joblib` = $1.73\text{ MB}$) and conformal calibrators. Move Random Forest baseline binaries to a GitHub Release asset.
  - *Tradeoff:* Reduces clone size to ~50 MB, but requires external asset downloading for baseline re-evaluation.

---

## 3. Recommended Phased Action Plan (Post-Approval)

Upon receiving user confirmation, the cleanup will execute in the following exact sequence:

1. **Phase 1: Create GitHub Configuration Files:**
   - Add `.gitignore` to prevent any future tracking of `__pycache__`, `.pytest_cache`, `.env`, and OS metadata.
   - Add `requirements.txt` specifying exact, pinned dependency ranges compatible with Python 3.10+.
2. **Phase 2: Purge Candidate Extraneous Files:**
   - Remove `data.csv`, `test.csv`, `2A2478DC-8517-4490-9FA7-36F9A7A542BE.JPG`.
   - Remove legacy root files `audit_dataset.py` and `audit_report.json`.
   - Remove redundant `PRSA2017_Data_20130301-20170228.zip` (retaining extracted CSVs).
   - Clean `__pycache__` and `.pytest_cache` directories.
3. **Phase 3: Sanitize Markdown Documentation Links:**
   - Replace local `file:///c:/Users/Dell/Downloads/...` links in `README.md` and `research/*.md` with clean GitHub-compatible relative paths (e.g. `[eda_report.md](research/eda_report.md)`).
4. **Phase 4: Verification of Clean State:**
   - Run `pytest tests/ -v` to ensure 100% test pass rate (52/52).
   - Test FastAPI and Streamlit startup.
