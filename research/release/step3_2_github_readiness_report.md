# AirCast Research — Step 3.2: Final Pre-GitHub Release Audit Report

**Role:** Senior ML Systems Engineer & Independent Release Auditor  
**Date:** November 2026  
**Audited Directory:** `beijing+multi+site+air+quality+data/`  
**Evaluation Scope:** Codebase Completeness, Dependency Mapping, Licensing Integrity, Security Hygiene, Large File Compliance, Scientific Precision, and Automated Test Regressions  
**Status:** **READY FOR GITHUB RELEASE (Pending User License Attribution Confirmation)**  

---

## 1. Executive Summary

This audit serves as the final technical gateway prior to Git initialization, local repository staging, and remote GitHub release. Every source code module, configuration artifact, trained model, conformal calibrator, documentation link, and dependency specification has been rigorously inspected against production open-source standards.

- **Automated Regression Suite:** **52 / 52 Unit Tests Passed (100.0% Pass Rate)** in $37.79\text{s}$.
- **Link Resolution:** **32 / 32 local Markdown links in `README.md` verified 100% valid**.
- **Model & Calibrator Verification:** All 3 recommended Pareto forecasters and 3 conformal calibrators exist, load cleanly, and execute predictions.
- **GitHub Storage Limits:** **Zero files exceed 50 MB or 100 MB** (largest file is `random_forest_h24.joblib` at $41.86\text{ MB}$). Standard Git push will succeed without requiring Git LFS.
- **Security & Secrets:** Automated regex scanning confirmed **zero API keys, passwords, private tokens, or personal identifiers**.
- **Scientific Honesty:** Atmospheric interpretations are strictly framed as observational associations and meteorological hypotheses. All sensor perturbation references use *"controlled sensor-perturbation evaluations"*.

---

## 2. Comprehensive Pre-Release Audit Matrix

| Audit Dimension | Specific Verification Check | Target / Threshold | Result / Value | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Scientific Wording** | Atmospheric SHAP interpretation | Association / Hypothesis only | Causal phrasing replaced | **PASSED** |
| **Scientific Wording** | Sensor stress testing phrasing | *"controlled sensor-perturbation"* | Standardized across all docs | **PASSED** |
| **License Formatting** | MIT License in root `LICENSE` | Standard legal text | Fully compliant | **PASSED** |
| **License Attribution** | Copyright holder & year | Explicit attribution | `2026 AirCast Research Team` | **PASSED (Actionable)** |
| **Third-Party Data** | PRSA 2017 dataset ownership | Zero claim of ownership | Explicitly attributed to Song Chen et al. | **PASSED** |
| **Source Files** | Core package modules | `src/`, `api/`, `dashboard/` | All 22 source modules present | **PASSED** |
| **Dependencies** | `requirements.txt` vs actual imports | Zero unlisted imports | `psutil` added; 100% covered | **PASSED** |
| **Dataset Setup** | PRSA 2017 raw files & UCI docs | Clear, reproducible setup | 12 CSVs pre-bundled (31 MB) + UCI link | **PASSED** |
| **Model Artifacts** | Pareto suite disk existence | `artifacts/models/` | All 6 recommended files exist | **PASSED** |
| **Model Validation** | Clean loading & inference execution | No exceptions on load | All 3 models + 3 calibrators verified | **PASSED** |
| **GitHub File Limits** | Max file size vs 100 MB hard limit | $\le 100\text{ MB}$ | Largest: $41.86\text{ MB}$ | **PASSED** |
| **GitHub LFS Threshold** | Max file size vs 50 MB warning | $\le 50\text{ MB}$ | Largest: $41.86\text{ MB}$ (No LFS needed) | **PASSED** |
| **Total Repo Size** | Total repository disk footprint | $< 250\text{ MB}$ | $185.05\text{ MB}$ (192 clean files) | **PASSED** |
| **Security & Privacy** | Private keys, tokens, personal paths | 0 credentials / local URLs | Clean (0 secrets found) | **PASSED** |
| **Documentation Links**| Local relative Markdown links | 100% resolution | 32 / 32 links valid | **PASSED** |
| **Regression Testing** | Automated pytest suite | 52 / 52 passing | 52 passed, 0 failed (37.79s) | **PASSED** |

---

## 3. Detailed Audit Findings by Task

### 3.1 Scientific Wording & Atmospheric Hypothesis Review
- **SHAP Feature Attributions:** Audited all references to TreeSHAP feature attributions in `README.md`, `research/release/step3_final_accuracy_review.md`, and technical reports. All statements regarding northerly wind ventilation, southerly advection, hygroscopic dew point depression, and winter heating baseline shifts are explicitly framed as **statistical associations consistent with domain meteorological hypotheses**, rather than causal proof.
- **Controlled Sensor-Perturbation Nomenclature:** Replaced all occurrences of the phrase *"causal evaluations"* or *"causal perturbation findings"* with *"controlled sensor-perturbation evaluations"*, ensuring methodological precision.
- **Zero New Claims:** No new scientific assertions or unverified hypotheses were introduced.

### 3.2 Licensing & Attribution Audit
- **Root `LICENSE` File:** Formatted according to standard open-source MIT License terms.
  - Current Header: `Copyright (c) 2026 AirCast Applied Sciences Research Team`
  - Recommendation for Approval: If the repository should reflect individual personal authorship rather than a research group entity, the user should approve updating the line to:
    ```text
    Copyright (c) 2026 [Author Name / GitHub Username]
    ```
    Per strict instructions, **no silent modification was made to the copyright line**.
- **Third-Party Data Attribution:** Explicitly documented in `README.md` Section 18 and audit reports that the PRSA 2017 dataset is the intellectual work of Song Chen et al. (Tsinghua / Peking University), published in *Proc. R. Soc. A (2017)* and distributed via the UCI Machine Learning Repository. No claim of ownership is made.
- **Third-Party Software Attribution:** Python dependencies (NumPy, PyTorch, XGBoost, FastAPI, Streamlit, etc.) are acknowledged under their respective BSD, Apache 2.0, and MIT licenses.

### 3.3 GitHub Repository Readiness & Dependency Audit
- **Source Code Verification:** All core scientific modules in `src/` (data, features, models, uncertainty, explainability, evaluation), serving endpoints in `api/`, interactive dashboards in `dashboard/`, reproducible data prep in `scripts/`, and unit tests in `tests/` are present and functional.
- **Dependency Consistency (`requirements.txt`):** AST scanning of all 22 Python source files detected 15 external third-party packages. `psutil>=5.9.0` (used for memory profiling and latency benchmarking in `scripts/`) was added to `requirements.txt`. Every single Python import is now declared with tested version bounds compatible with Python 3.13.5.
- **Dataset Setup & Onboarding Clarity:**
  - The repository includes all 12 raw station telemetry CSV files ($31.00\text{ MB}$) in `PRSA_Data_20130301-20170228/`.
  - The preprocessed Parquet cache ($30.52\text{ MB}$, 420,768 rows $\times$ 70 columns) is pre-bundled in `data/processed/forecasting_features_cache.parquet`.
  - A direct link to the [UCI Machine Learning Repository: Beijing Multi-Site Air-Quality Data](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) has been added to `README.md` for independent acquisition.
  - A single command (`python scripts/prepare_processed_data.py`) can regenerate the entire feature cache from raw CSVs in $\sim 28\text{ seconds}$.
- **Large File Analysis & Git LFS Verification:**
  - Hard Limit Check ($100\text{ MB}$): **0 files exceed 100 MB**.
  - Warning Threshold Check ($50\text{ MB}$): **0 files exceed 50 MB**.
  - Top 4 Largest Files on Disk:
    1. `artifacts/models/random_forest_h24.joblib`: **$41.86\text{ MB}$**
    2. `artifacts/models/random_forest_h6.joblib`: **$34.22\text{ MB}$**
    3. `data/processed/forecasting_features_cache.parquet`: **$30.52\text{ MB}$**
    4. `artifacts/models/random_forest_h1.joblib`: **$24.59\text{ MB}$**
  - **Verdict:** Git LFS is **NOT REQUIRED**. A standard `git push` to GitHub will succeed without warnings or blocks.
- **Security, Secrets, & Privacy Hygiene:**
  - Regex scans across all source files, documentation, and configuration files confirmed **zero credentials, API tokens, passwords, or cloud secrets**.
  - Personal local paths (`file:///c:/Users/Dell/...`) were stripped from documentation and converted to clean repository-relative links.
  - Temporary files, OS metadata (`Thumbs.db`), and bytecode directories (`__pycache__`, `.pytest_cache`) are excluded via `.gitignore`.

### 3.4 Verification of Production Artifacts & Automated Tests
- **Model Artifact Validation:**
  - `artifacts/models/xgboost_h1.joblib` ($1.1577\text{ MB}$, 180 trees, max depth 7): Verified.
  - `artifacts/models/xgboost_pruned_h6.joblib` ($0.4328\text{ MB}$, 90 trees, max depth 6): Verified.
  - `artifacts/models/xgboost_fast_h24.joblib` ($0.1352\text{ MB}$, 50 trees, max depth 5): Verified.
- **Conformal Calibrator Validation:**
  - `artifacts/models/conformal_calibrator_h1.joblib` ($1.1743\text{ MB}$): Verified.
  - `artifacts/models/conformal_calibrator_h6.joblib` ($1.2534\text{ MB}$): Verified.
  - `artifacts/models/conformal_calibrator_h24.joblib` ($0.8918\text{ MB}$): Verified.
- **Automated Regression Suite:**
  ```text
  pytest tests/ -q
  .................................................... [100%]
  52 passed, 6 warnings in 37.79s
  ```
  **Result: 52 Passed, 0 Failed.**

---

## 4. Summary of Checks

### 4.1 Passed Checks (16 / 16)
1. Exact path-dependent TreeSHAP attributions framed strictly as associations/hypotheses.
2. Controlled sensor-perturbation terminology applied across all documentation.
3. Root `LICENSE` file present and formatted with standard MIT terms.
4. Third-party dataset attribution clearly delineated without ownership claims.
5. All required source packages (`src/`, `api/`, `dashboard/`, `scripts/`, `tests/`) present.
6. `requirements.txt` updated to include `psutil` alongside all 14 core dependencies.
7. Dataset download and cache regeneration instructions verified and reproducible.
8. Model artifact paths referenced in `README.md` verified on disk.
9. Recommended models (`xgboost_h1`, `xgboost_pruned_h6`, `xgboost_fast_h24`) validated.
10. Conformal calibrators (`h1`, `h6`, `h24`) validated.
11. Zero files exceed GitHub's 100 MB block limit.
12. Zero files exceed GitHub's 50 MB warning threshold (no Git LFS needed).
13. Total repository size is $185.05\text{ MB}$ across 192 clean files.
14. Zero credentials, tokens, passwords, or personal file paths present.
15. 32 of 32 local Markdown links in `README.md` resolve successfully.
16. Pytest test suite executed: 52 of 52 tests passing (100.0% pass rate).

### 4.2 Failed Checks (0 / 16)
- **None.** Zero technical or compliance failures detected.

### 4.3 Warnings & Items for User Confirmation
- **Action Item (Optional):** In [`LICENSE`](../../LICENSE), the copyright line currently reads:
  ```text
  Copyright (c) 2026 AirCast Applied Sciences Research Team
  ```
  If you prefer this portfolio repository to carry your individual name or GitHub handle (e.g., `Copyright (c) 2026 [Your Name]`), please provide your preferred name and approval to update it.

---

## 5. Final Recommendation

# **RECOMMENDATION: READY FOR GITHUB RELEASE**

The AirCast Research repository is in an exemplary, publication-grade state:
- All scientific claims are defensible and substantiated by primary JSON metrics.
- All code runs cleanly on commodity CPU hardware with Python 3.13.5.
- The repository structure is under 186 MB, clean of temporary files, and strictly below all GitHub size thresholds.
- Zero Git actions have been taken (no `.git` initialization, no staging, no commits, no remotes, no pushes).

**Awaiting your approval regarding the license author attribution before proceeding to Step 4 (Git Initialization & Initial Release Commit).**
