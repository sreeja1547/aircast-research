# AirCast Research — Step 2 Cleanup & GitHub Preparation Execution Report

**Auditor Role:** Senior ML Systems Architect & Independent Open-Source Release Auditor  
**Date:** November 2026  
**Status:** Step 2 Cleanup Executed, Verified, and Checked — STOPPED Prior to Remote Repository Initialization  
**Primary Tested Runtime:** Python 3.13.5 on Windows 11 Home x64  
**Backup Location:** `C:\Users\Dell\.gemini\antigravity\brain\19556efc-88fd-4fd7-ab89-229d036a68e7\scratch\cleanup_backup\`

---

## 1. Executive Summary & Cleanup Impact

Step 2 cleanup has been executed in strict adherence to the non-negotiable governance rules:
- **Zero Scientific Regression:** No models were retrained, no hyperparameters were altered, and no ML features were modified.
- **Evidence Preserved:** All 26 metrics JSON files, 21 publication plots, and 30 research reports remain intact.
- **Reversibility Guaranteed:** Every removed file was verified, hashed (SHA-256), and backed up outside the repository before removal.
- **Repository Optimization:** File count decreased from **238 to 189 files**, and repository disk footprint was reduced from **193.26 MB to 185.02 MB** (a reduction of 8.24 MB of redundant/unrelated artifacts).
- **Test Integrity:** All 52 automated regression tests pass (100.0% pass rate in 41.18s), and all 26 internal documentation links in `README.md` resolve to valid files.

---

## 2. Inventory of Removed & Backed-Up Files

All removal candidates were hashed and copied to the external backup directory before deletion from the project root:

| File Name | Original Size | SHA-256 Checksum | Classification & Rationale | Backup Status |
| :--- | :---: | :--- | :--- | :---: |
| `data.csv` | 37,769 B | `cfd747b9f7308e6e5ec84944f963bbb2ca39a90e1d235c0f1e2c565afed88b3d` | Unrelated 2018 stock price data (`Open,High,Low,Close,Volume`). | Backed Up & Removed |
| `test.csv` | 35,186 B | `0a1478eb69615e3673601dfd028251898faae73c73f0cbb0b7653b26c8b74b9e` | Unrelated 2018 stock price data (`Open,High,Low,Close,Volume`). | Backed Up & Removed |
| `2A2478DC-8517-4490-9FA7-36F9A7A542BE.JPG` | 158,758 B | `b18003e2f3612d683daea2c623e65327e7a2fab63f0ff23f7b3cbda1a704290a` | Standalone JPEG from May 2023. Unreferenced in project. | Backed Up & Removed |
| `PRSA2017_Data_20130301-20170228.zip` | 7,959,991 B | `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8` | Redundant archive. All 12 station CSVs verified bit-identical in `PRSA_Data_20130301-20170228/`. | Backed Up & Removed |
| `audit_dataset.py` | 8,804 B | `a4884166721f409b11f2251830401aabb8c52e6085ca391665a31be2fded6f05` | Legacy Phase 1 audit script. Fully superseded by `src/data/loader.py` and pipeline. | Backed Up & Removed |
| `audit_report.json` | 23,300 B | `37dbc22bfe8599e3d97fe4169cc3b248c177d9e26806c57a8f2f8251f8910157` | Legacy Phase 1 audit JSON. Fully superseded by `eda_statistical_summary.json`. | Backed Up & Removed |
| `**/__pycache__/*.pyc` (12 dirs) | ~350 KB | — | Compiled Python bytecode. Ephemeral. | Purged |
| `.pytest_cache/` | ~15 KB | — | Test runner cache. Ephemeral. | Purged |

### Verification of Extracted Raw Dataset Integrity
Before removing `PRSA2017_Data_20130301-20170228.zip`, each internal CSV stream was extracted in-memory and compared against `PRSA_Data_20130301-20170228/`:
- `PRSA_Data_Aotizhongxin_20130301-20170228.csv` (2,835,916 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Changping_20130301-20170228.csv` (2,722,295 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Dingling_20130301-20170228.csv` (2,675,856 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Dongsi_20130301-20170228.csv` (2,636,684 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Guanyuan_20130301-20170228.csv` (2,695,860 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Gucheng_20130301-20170228.csv` (2,654,625 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Huairou_20130301-20170228.csv` (2,641,027 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Nongzhanguan_20130301-20170228.csv` (2,839,705 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Shunyi_20130301-20170228.csv` (2,620,654 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Tiantan_20130301-20170228.csv` (2,655,061 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Wanliu_20130301-20170228.csv` (2,659,544 B): **SHA-256 Match: TRUE**
- `PRSA_Data_Wanshouxigong_20130301-20170228.csv` (2,871,076 B): **SHA-256 Match: TRUE**
*All 12 raw station files remain fully intact and active in `PRSA_Data_20130301-20170228/`.*

---

## 3. Retained Core Assets Inventory

The post-cleanup repository houses **189 clean, verified files**:

1. **Source Packages (`src/`):** 22 modules across `data`, `evaluation`, `experiments`, `explainability`, `features`, `models`, `preprocessing`, and `uncertainty`.
2. **Serving Layer (`api/`):** `main.py`, `schemas.py`, and `services.py` providing FastAPI endpoints.
3. **Research Dashboard (`dashboard/`):** `app.py` providing the 9-tab Streamlit visual interface.
4. **Automated Test Suite (`tests/`):** 9 test files covering all functionality.
5. **Recommended Deployment Models:**
   - `artifacts/models/xgboost_h1.joblib` (1.1577 MB, 180 trees, depth 7)
   - `artifacts/models/xgboost_pruned_h6.joblib` (0.4328 MB, 90 trees, depth 6)
   - `artifacts/models/xgboost_fast_h24.joblib` (0.1352 MB, 50 trees, depth 5)
6. **Normalized Conformal Calibrators:**
   - `artifacts/models/conformal_calibrator_h1.joblib` (1.1743 MB)
   - `artifacts/models/conformal_calibrator_h6.joblib` (1.2534 MB)
   - `artifacts/models/conformal_calibrator_h24.joblib` (0.8918 MB)
7. **Baseline Research Binaries:** All 26 unpruned baseline models (Random Forest, HistGBDT, LSTM, Ridge, Linear Regression, Persistence, Standard Conformal) remain preserved in `artifacts/models/`.
8. **Processed Dataset Cache:** `data/processed/forecasting_features_cache.parquet` (30.52 MB, 420,768 rows $\times$ 70 columns).
9. **Research Reports & Metrics:** 30 comprehensive markdown reports in `research/` and 26 primary JSON files in `artifacts/metrics/`.

---

## 4. Configuration & Dependency Files Created

### 4.1 `.gitignore` Implementation
Created at the repository root to enforce strict hygiene for Git tracking:
- Excludes: `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `venv/`, `.venv/`, `.env`, IDE files (`.vscode/`, `.idea/`), OS files (`Thumbs.db`, `.DS_Store`), logs, and scratch scripts.
- Explicitly whitelists: `!artifacts/models/*.joblib`, `!artifacts/metrics/*.json`, `!artifacts/plots/*.png`, `!data/processed/forecasting_features_cache.parquet`, `!PRSA_Data_20130301-20170228/*.csv`, and `!research/**/*.md`.

### 4.2 `requirements.txt` Implementation
Created with tested, bounded dependency declarations:
- **Tested Python Version:** Python 3.13.5 (Windows 11 x64).
- **Core Packages:** `numpy>=1.24.0,<2.3.0`, `pandas>=2.0.0,<3.1.0`, `scipy>=1.10.0,<1.18.0`, `pyarrow>=12.0.0`, `joblib>=1.3.0`.
- **Machine Learning:** `scikit-learn>=1.3.0`, `xgboost>=2.0.0`, `torch>=2.0.0` (CPU-only build verified).
- **Serving & UI:** `fastapi>=0.100.0`, `uvicorn>=0.22.0`, `pydantic>=2.0.0`, `streamlit>=1.28.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`.
- **Testing:** `pytest>=7.4.0`.
- *Omission Justification:* The external `shap` package was purposefully excluded because `src/explainability/shap_engine.py` leverages native XGBoost C++ TreeSHAP (`pred_contribs=True`), eliminating external compilation overhead.

---

## 5. Documentation Link Sanitization

- Scanned all 33 markdown documentation files across the workspace.
- **52 absolute Windows URLs** (`file:///c:/Users/Dell/Downloads/beijing+multi+site+air+quality+data/...`) were identified and converted to clean repository-relative Markdown links.
- Verified that **26 out of 26 links in `README.md`** resolve to actual, existing files on disk.
- Zero local `file:///` links remain active in any document.

---

## 6. GitHub Storage & Security Audit

### 6.1 Storage Breakdown Before vs After
- **Total Repository Footprint:** Reduced from **193.26 MB** to **185.02 MB** ($-8.24\text{ MB}$, $-4.26\%$).
- **Total Trackable Files:** Reduced from **238** to **189 files** ($-49$ files including pycache purge).

### 6.2 Large File Analysis
```
File Path                                            File Size   GitHub Status
----------------------------------------------------------------------------------------
artifacts/models/random_forest_h24.joblib            41.86 MB    PASS (< 50 MB warning)
artifacts/models/random_forest_h6.joblib             34.22 MB    PASS (< 50 MB warning)
data/processed/forecasting_features_cache.parquet    30.52 MB    PASS (< 50 MB warning)
artifacts/models/random_forest_h1.joblib             24.59 MB    PASS (< 50 MB warning)
```
- **Files $\ge 10\text{ MB}$:** Exactly 4 files.
- **Files $\ge 50\text{ MB}$ (GitHub Warning):** **0 files.**
- **Files $\ge 100\text{ MB}$ (GitHub Block):** **0 files.**
- Standard `git push` will execute smoothly without requiring Git LFS or encountering size blocks.

### 6.3 Security & Personal Information Scan
- Zero API keys, passwords, bearer tokens, or cloud secrets exist in the codebase.
- No personal user paths (`Users/Dell`) exist in any Python source file.
- Git is currently uninitialized (`.git` directory absent); no commits or remotes have been created.

---

## 7. Verification Results (Post-Cleanup)

| Verification Gate | Command / Target | Result | Status |
| :--- | :--- | :--- | :---: |
| **Pytest Suite** | `pytest tests/ -v` | **52 passed, 0 failed, 6 warnings in 41.18s** | **PASS (100%)** |
| **Model Loading** | `joblib.load(artifacts/models/xgboost_*.joblib)` | All 3 Pareto models load and validate | **PASS** |
| **Calibrator Loading**| `ConformalPredictor.load(artifacts/models/conformal_calibrator_*.joblib)` | All 3 normalized engines load and validate | **PASS** |
| **Inference Service**| `AirCastInferenceService.predict_single()` | Real-time prediction and interval generated across +1h, +6h, +24h | **PASS** |
| **FastAPI Routes** | `api.main.app` route registration | All 8 routes registered (`/health`, `/model-info`, `/predict`, etc.) | **PASS** |
| **Dashboard Import**| `import dashboard.app` | Streamlit helpers and UI modules import cleanly | **PASS** |
| **README Links** | Check all relative Markdown links | 26 / 26 links point to existing files | **PASS (100%)** |
| **Dataset Integrity**| `load_all_stations()`, `df_cache.parquet` | 12 stations verified; 420,768 rows $\times$ 70 cols confirmed | **PASS** |

---

## 8. Conclusion & Stop Gate Notice

Step 2 cleanup and preparation is complete. All redundant files are safely backed up outside the repository, all documentation links are sanitized, `.gitignore` and `requirements.txt` are created, and all verification tests pass at 100%.

> [!IMPORTANT]
> **AUDITOR STOP GATE:**  
> The auditor has stopped execution. No Git repository has been initialized, no commits have been created, and no remote connections have been opened. Awaiting user instructions.
