# AirCast Research: Stage 0 — Computational Environment & System Audit

**Phase 11: Computational Efficiency, Model Optimization & Inference Acceleration**  
**Author:** Principal ML Systems & Performance Engineer  
**Timestamp:** 2026-11-05T20:20:00+05:30

---

## 1. Executive Summary

As the foundational step of Phase 11, this audit inspects the physical execution environment, software dependency stack, model artifacts, data caching mechanisms, and end-to-end inference paths of the AirCast Research framework. 

All metrics documented below were measured directly on the local workstation to eliminate assumptions and establish an empirical foundation for performance optimization.

---

## 2. Hardware Environment Audit

Direct hardware inspection via `psutil`, `platform`, `torch`, and `xgboost` diagnostics:

| Parameter | Measured Specification | Engineering Implication |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Pro (10.0.26300-SP0) | Standard multi-threaded Windows process management |
| **Processor Architecture** | Intel64 Family 6 Model 186 Stepping 3, GenuineIntel | Modern Intel hybrid architecture |
| **Physical CPU Cores** | 10 Physical Cores | High thread-level parallelism for multi-threaded tree ensembles |
| **Logical CPU Cores** | 12 Logical Cores (Hyperthreads) | Optimal worker concurrency target for `n_jobs` and OpenMP |
| **Nominal Base Frequency** | 1,300 MHz (Dynamic Turbo scaling) | Memory-bandwidth and cache-hierarchy sensitivity |
| **Total System RAM** | 15.69 GB (~16.0 GB physical) | Strict ceiling on in-memory DataFrame duplication |
| **Available RAM at Audit** | **2.76 GB** | **High memory pressure:** large uncompressed arrays risk paging |
| **GPU Hardware Status** | Discrete GPU not visible to Python runtime | `nvidia-smi` is not exposed in system runtime path |
| **PyTorch Device Build** | `2.10.0+cpu` (`torch.cuda.is_available() == False`) | All PyTorch LSTM tensor operations execute strictly on CPU |
| **XGBoost Device Support** | Falls back to CPU (`context.cc: No visible GPU found`) | All histogram tree boosting executes strictly on CPU |

> [!IMPORTANT]
> **Strict Empirical Grounding (Principle 18):** Because the runtime environment utilizes a CPU-only PyTorch build (`2.10.0+cpu`) and XGBoost reports no visible CUDA device, **no GPU acceleration will be claimed or fabricated**. All training and inference optimization must focus on CPU vectorization, memory locality, algorithmic pruning, thread scheduling, and data structures.

---

## 3. Software Dependency Stack

| Software Package | Installed Version | Primary Role |
| :--- | :---: | :--- |
| **Python** | 3.13.5 | Runtime environment |
| **NumPy** | 2.2.6 | Array computation & BLAS backend |
| **Pandas** | 3.0.0 | Tabular data manipulation & Parquet I/O |
| **Scikit-Learn** | 1.9.0 | Linear models, HistGBDT, Random Forest, metrics |
| **XGBoost** | 3.4.1 | Optimized C++ histogram tree boosting |
| **PyTorch** | 2.10.0+cpu | Recurrent neural networks (LSTM) |
| **FastAPI** | 0.115.14 | High-performance asynchronous REST API |
| **Streamlit** | 1.45.0 | Multi-panel scientific research dashboard |
| **Joblib** | 1.4.2 | Model serialization & multi-process pipelines |
| **Pytest** | 8.3.4 | Automated regression verification suite (47 tests) |

---

## 4. Model Artifacts & Storage Footprint Audit

A comprehensive scan of `artifacts/models/` reveals a total disk consumption of **111.45 MB** across 24 serialized forecaster and conformal calibrator artifacts:

| Architecture Family | File Pattern | Count | Individual File Size | Subtotal Size (MB) | Share (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | `random_forest_h{1,6,24}.joblib` | 3 | $24.59 - 41.86\text{ MB}$ | **100.67 MB** | **90.3%** |
| **XGBoost** | `xgboost_h{1,6,24}.joblib` | 3 | $0.57 - 1.16\text{ MB}$ | **2.81 MB** | **2.5%** |
| **HistGradientBoosting** | `hist_gradient_boosting_h{1,6,24}.joblib` | 3 | $0.58\text{ MB}$ | **1.74 MB** | **1.6%** |
| **PyTorch LSTM** | `lstm_h{1,6,24}.joblib` | 3 | $0.25\text{ MB}$ | **0.75 MB** | **0.7%** |
| **Conformal Calibrators** | `conformal_calibrator_h{1,6,24}.joblib` | 3 | $0.89 - 1.25\text{ MB}$ | **3.31 MB** | **3.0%** |
| **Standard Calibrators** | `conformal_calibrator_std_h{1,6,24}.joblib` | 3 | $0.89 - 1.25\text{ MB}$ | **3.31 MB** | **3.0%** |
| **Linear / Ridge / Persist** | `{linear,ridge,persistence}_h*.joblib` | 9 | $< 0.005\text{ MB}$ | **0.02 MB** | **<0.1%** |
| **Total Models** | | **27** | | **112.61 MB** | **100.0%** |

### Critical Storage Finding:
Random Forest accounts for **90.3%** of all model artifact storage ($41.86\text{ MB}$ for $h=24\text{h}$ alone). Each tree is unpruned with up to 16 depth levels across 60–100 estimators, storing hundreds of thousands of decision split nodes in uncompressed Python object structures.

---

## 5. Processed Dataset & In-Memory Footprint

| Data Asset | Path | File Size on Disk | In-Memory DataFrame RAM | Rows | Columns |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Causal Parquet Cache** | `data/processed/forecasting_features_cache.parquet` | **30.52 MB** | **203.31 MB** | 420,768 | 70 |

* **Parquet Efficiency:** Snappy-compressed Parquet provides a $6.66\times$ compression ratio relative to uncompressed in-memory representation ($30.52\text{ MB}$ vs. $203.31\text{ MB}$).
* **Data Types:** The cache currently stores all numerical features as 64-bit floating-point (`float64`) and integer (`int64`) types. 
* **Optimization Potential:** Downcasting sensor channels, rolling statistics, and cyclical harmonics to 32-bit floating-point (`float32`) can reduce memory by approximately $50\%$ (~$100\text{ MB}$ saved per DataFrame copy) with negligible numerical loss.

---

## 6. End-to-End Serving Pipeline Audit

### A. FastAPI REST API (`api/services.py`, `api/main.py`)
1. **Single-Sample Inference Path:**
   - Input: Pydantic dict of 47 feature key-values.
   - Conversion: Builds 1-row Pandas DataFrame `df_sample = pd.DataFrame([row_dict])`.
   - Validation Overhead: Calls `model._validate_inputs()`, which re-checks columns, copies DataFrame `X_out = X_out[self.feature_names].copy()`, and iterates over categorical mappings.
   - Prediction: Model executes `.predict()` on DataFrame.
   - Interval: Calibrator predicts intervals, slicing DataFrame.
2. **Serving Bottleneck:** Converting Python dictionary to Pandas DataFrame and immediately back to NumPy array inside scikit-learn/XGBoost adds 1–3 ms of pure Python wrapper overhead per single-sample prediction.

### B. Streamlit Research Dashboard (`dashboard/app.py`)
1. **Uncached File I/O:** `load_json_safe()` is invoked directly inside tab render functions without `@st.cache_data`. Every user click or tab switch re-reads multiple metric JSON files from the filesystem.
2. **Tab 9 Mock Execution:** Tab 9 ("Interactive Live Forecaster") previously used a heuristic formula rather than the unified `AirCastInferenceService`, causing slight behavioral divergence from the production API.

---

## 7. Baseline Verification Status

The automated regression test suite (`tests/`) currently contains **47 passing tests** with 0 failures:
- `tests/test_api.py` (9 tests)
- `tests/test_conformal.py` (3 tests)
- `tests/test_evaluation.py` (5 tests)
- `tests/test_explainability.py` (1 test)
- `tests/test_feature_engineering.py` (13 tests)
- `tests/test_models.py` (7 tests)
- `tests/test_preprocessing.py` (7 tests)
- `tests/test_robustness.py` (2 tests)

All 47 tests must remain 100% functional after optimizations.
