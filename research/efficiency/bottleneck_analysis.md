# AirCast Research: Stage 0 — Computational Bottleneck & Inefficiency Analysis

**Phase 11: Computational Efficiency, Model Optimization & Inference Acceleration**  
**Author:** Principal ML Systems & Performance Engineer  
**Timestamp:** 2026-11-05T20:22:00+05:30

---

## 1. Overview & Profiling Taxonomy

This document catalogs every computational bottleneck, memory allocation inefficiency, and architectural redundancy identified during the comprehensive audit of the AirCast Research framework. 

Each bottleneck is categorized across five performance axes:
1. **Feature Engineering & Dimensionality**
2. **Model Architecture Complexity**
3. **Training Pipeline & Data Pipeline**
4. **Online Inference & Serving Latency**
5. **Memory Management & Storage Footprint**

---

## 2. Identified Computational Bottlenecks

### Bottleneck 1: Universal 47-Feature Dimensionality Across All Horizons
* **Location:** `src/features/registry.py`, `src/experiments/benchmarking.py`
* **Root Cause:** All 7 forecaster architectures across all 3 lead times ($1\text{h}, 6\text{h}, 24\text{h}$) train and predict on the complete 47-feature `combined_all` group.
* **Inefficiency:** 
  - For $h=1\text{h}$, immediate particulate inertia ($PM_{2.5}, PM_{10}$, short lags) accounts for $>80\%$ of predictive power. 48-hour rolling statistics and remote trace gas harmonics add marginal value while forcing every tree node to evaluate 47 split candidates.
  - In random forests, split search time scales linearly with $p$ (number of features). Reducing feature dimensionality directly reduces training and inference latency across all tree models.

---

### Bottleneck 2: Massive Unpruned Decision Trees in Random Forest
* **Location:** `src/models/trees.py` (`RandomForestForecaster`)
* **Root Cause:** Random Forest models are initialized with `n_estimators=60-100`, `max_depth=14-16`, and `min_samples_leaf=4`.
* **Inefficiency:**
  - Artifact files range from **$24.59\text{ MB}$ to $41.86\text{ MB}$** per horizon ($100.67\text{ MB}$ total).
  - High tree depth creates massive pointer structures in memory, causing poor CPU L1/L2 cache locality during tree traversal.
  - Single-sample inference requires traversing 60 deep decision trees sequentially or across thread pools, resulting in high tail latency (P95 > 15 ms).

---

### Bottleneck 3: Redundant DataFrame Allocation in Input Validation
* **Location:** `src/models/base.py` (`_validate_inputs`)
* **Root Cause:**
  ```python
  # In _validate_inputs(is_training=False):
  X_out = X_out[self.feature_names].copy()
  for col, mapping in self.categorical_encoders_.items():
      if col in X_out.columns:
          X_out[col] = X_out[col].astype(str).map(mapping).fillna(-1.0).astype(np.float64)
  ```
* **Inefficiency:**
  - Every single call to `.predict()` allocates a fresh copy of the DataFrame `X_out[self.feature_names].copy()`.
  - For high-throughput batch inference or repeated single-sample REST queries, memory allocation and string conversion dominate the actual matrix multiplication or tree split evaluation.

---

### Bottleneck 4: 64-Bit Precision Overhead (`float64`) Across Entire Pipeline
* **Location:** `data/processed/forecasting_features_cache.parquet`, `src/data/forecasting_dataset.py`
* **Root Cause:** Numerical features (sensor readings, wind vectors, rolling statistics, lags) are stored as double-precision `float64` (8 bytes per value).
* **Inefficiency:**
  - Environmental sensors have at most 1–2 decimal digits of physical precision (e.g., $PM_{2.5} = 45.2\ \mu\text{g}/\text{m}^3$). Double-precision floats provide 53 bits of mantissa (~15-17 decimal digits), wasting $50\%$ of RAM bandwidth and cache space.
  - In-memory cache consumes **$203.31\text{ MB}$** in `float64`; downcasting to single-precision `float32` (4 bytes per value) will reduce memory consumption to **~$102\text{ MB}$** with zero measurable impact on RMSE or MAE.

---

### Bottleneck 5: Uncached Dashboard I/O on UI Interactions
* **Location:** `dashboard/app.py`
* **Root Cause:** Helper function `load_json_safe(path)` reads JSON files directly from disk on every Streamlit script rerun:
  ```python
  def load_json_safe(path: str):
      if os.path.exists(path):
          with open(path, "r", encoding="utf-8") as f:
              return json.load(f)
  ```
* **Inefficiency:**
  - Streamlit executes the entire script top-to-bottom on every user widget event (e.g., slider movement, tab selection).
  - Re-reading 8+ JSON metric files on every rerun introduces disk I/O latency and UI stutter. Applying Streamlit's `@st.cache_data` will keep deserialized JSON structures pinned in RAM.

---

### Bottleneck 6: Single-Sample REST API Object Conversion Overhead
* **Location:** `api/services.py` (`predict_single`)
* **Root Cause:** 
  1. REST payload arrives as a Python dictionary (`Dict[str, float]`).
  2. `predict_single` constructs a 1-row Pandas DataFrame:
     `df_sample = pd.DataFrame([row_dict])`
  3. Model's `_validate_inputs` copies the DataFrame and extracts NumPy arrays.
  4. Scikit-learn/XGBoost converts the DataFrame back into an internal array.
* **Inefficiency:**
  - Slicing and indexing a 1-row Pandas DataFrame has an overhead of ~0.8–1.5 ms in pure Python.
  - Passing a pre-allocated 1D/2D NumPy array or contiguous buffer directly to the forecaster cuts single-sample overhead by $>60\%$.

---

### Bottleneck 7: PyTorch LSTM DataLoader Overhead on CPU
* **Location:** `src/models/deep_learning.py` (`LSTMForecaster`)
* **Root Cause:** PyTorch LSTM creates a standard `DataLoader` with default `batch_size=512`, `num_workers=0`, and standard tensor allocation on CPU.
* **Inefficiency:**
  - Single-threaded DataLoader batch generation creates CPU bubbles between mini-batch forward/backward passes.
  - Tuning mini-batch size and using direct tensor indexing during inference accelerates LSTM execution substantially.

---

## 3. Prioritized Optimization Roadmap

| Priority | Stage | Target Component | Expected Performance Benefit |
| :---: | :--- | :--- | :--- |
| **P0** | **Stage 1** | Baseline Benchmarking | Establish rigorous, repeatable measurement suite (RAM, latency, p95, throughput) |
| **P0** | **Stage 2** | Feature Selection & Pruning | Reduce feature set from 47 to optimal compact subset per horizon (target: $30\text{--}50\%$ fewer features) |
| **P0** | **Stage 3** | Model Complexity Tuning | Prune Random Forest depth/estimators; optimize XGBoost boosting rounds and early stopping |
| **P1** | **Stage 4** | Training Time Optimization | Vectorize preprocessing, float32 conversion, parallel thread utilization |
| **P1** | **Stage 5** | Inference Acceleration | Bypass DataFrame overhead in API, direct NumPy inference, vectorize batch prediction |
| **P1** | **Stage 6** | Memory & Storage Optimization | Float32 cache, compact artifact compression, RAM footprint reduction |
| **P2** | **Stage 7** | Horizon-Specific Selection | Evidence-based Pareto selection (best accuracy vs. efficiency per horizon) |
| **P2** | **Stage 8** | API & Dashboard Performance | Streamlit `@st.cache_data`, API throughput stress testing |
| **P0** | **Stage 9 & 10** | Regression & Final Report | Verify 47/47 passing tests, ensure zero accuracy loss, generate complete audit reports |
