# AirCast Research — Phase 11 Final Scientific and Performance Reconciliation Report

> **Auditor Role:** Senior Applied Scientist, Research Reviewer, ML Performance Engineer, and Statistical Systems Auditor  
> **Date:** November 2026  
> **Status:** Completed & Empirically Verified (Post-Phase 11 Audit)  
> **Hardware Environment:** Intel(R) Core(TM) i7-13700H CPU (14 cores / 20 threads, base 2.4 GHz, boost 5.0 GHz), 16.0 GB DDR5 RAM, Windows 11 Home (x64), Python 3.13.5, PyTorch 2.10.0 (CPU-only, `CUDA_AVAILABLE = False`).

---

## 1. Executive Summary & Audit Mandate

This report provides the definitive scientific reconciliation and empirical audit of Phase 11 (*Computational Efficiency, Model Optimization & Inference Acceleration*) for the AirCast Research multi-horizon atmospheric forecasting framework.

Rather than relying on narrative assertions or copied Markdown tables, this audit recomputed every reported metric directly from primary JSON artifacts (`artifacts/metrics/*.json`) and physical serialized models on disk (`artifacts/models/*.joblib`). Every identified discrepancy between intermediate stage reports has been investigated down to source code execution paths and resolved with full mathematical and procedural transparency.

### Summary of Final Reconciled Findings

| Metric Category | Baseline (Unoptimized) | Verified Optimized Result | Reconciled Performance Impact | Verification Source |
| :--- | :--- | :--- | :--- | :--- |
| **Target Deployment Footprint** | $100.67\text{ MB}$ (Baseline RF Suite) | $\mathbf{1.73\text{ MB}}$ (Pareto XGBoost Suite) | **$-98.29\%$ storage reduction ($58.33\times$ smaller)** | `artifacts/models/*.joblib` NTFS audit |
| **Full Model Archive Footprint** | $113.21\text{ MB}$ (29 physical artifacts) | $\mathbf{1.73\text{ MB}}$ (Recommended 3-model suite) | **$-98.48\%$ footprint reduction ($65.60\times$ smaller)** | `artifacts/models/*.joblib` NTFS audit |
| **Uniform XGBoost Footprint** | $2.81\text{ MB}$ (Baseline XGBoost Suite) | $\mathbf{1.73\text{ MB}}$ (Pruned/Fast Pareto Suite) | **$-38.54\%$ storage reduction ($1.63\times$ smaller)** | `phase11_final_verified_results.json` |
| **Active DataFrame RAM** | $227.38\text{ MB}$ (64-bit float) | $\mathbf{121.45\text{ MB}}$ (32-bit float downcast) | **$-46.59\%$ RAM reduction ($1.87\times$ smaller)** | `memory_optimization.json` |
| **Feature Extraction Speed** | $4.201\text{ s}$ (Iterative row windowing) | $\mathbf{0.0496\text{ s}}$ (Vectorized GroupBy) | **$84.70\times$ computational speedup** | `training_efficiency.json` |
| **C++ Buffer Inference Latency** | $35.60\text{ ms}$ (Pandas DataFrame path) | $\mathbf{4.33\text{ ms}}$ (Preallocated NumPy buffer) | **$8.22\times$ inference acceleration** (XGBoost) | `inference_optimization.json` |
| **REST API Serving Latency (P95)** | $71.86\text{ ms}$ (Initial unoptimized API) | $\mathbf{43.32\text{–}44.13\text{ ms}}$ (P95 across horizons) | **$1.63\times$ serving acceleration** (Unmet for 15 ms target) | Live 200-sample REST profile |
| **Batch Serving Throughput** | $286.6\text{ samples/s}$ ($b=1$ XGBoost) | $\mathbf{188,688.3\text{ samples/s}}$ ($b=1024$ XGBoost) | **$658.4\times$ throughput scaling** | `inference_optimization.json` |
| **Peak Linear Throughput** | $7,946.3\text{ samples/s}$ ($b=1$ Ridge) | $\mathbf{4,335,492.7\text{ samples/s}}$ ($b=1024$ Ridge) | **$545.6\times$ throughput scaling** | `inference_optimization.json` |
| **Horizon +1h Empirical Coverage**| Nominal Target: $90.0\%$ | $\mathbf{89.07\%}$ (Mean width: $44.88\ \mu\text{g}/\text{m}^3$) | Slight gap ($-0.93\%$); severe tier: $86.34\%$ | Live test evaluation ($N=42,554$) |
| **Horizon +6h Empirical Coverage**| Nominal Target: $90.0\%$ | $\mathbf{85.26\%}$ (Mean width: $131.29\ \mu\text{g}/\text{m}^3$) | Gap ($-4.74\%$); severe tier: $69.40\%$ | Live test evaluation ($N=42,494$) |
| **Horizon +24h Empirical Coverage**| Nominal Target: $90.0\%$ | $\mathbf{79.81\%}$ (Mean width: $181.15\ \mu\text{g}/\text{m}^3$) | Severe gap ($-10.19\%$); severe tier: $28.05\%$ | Live test evaluation ($N=42,278$) |
| **Pooled Conformal Test Coverage**| Nominal Target: $90.0\%$ | $\mathbf{84.72\%}$ (All horizons pooled, $N=127,326$) | Gap ($-5.28\%$); exchangeability violated in winter | Live pooled evaluation |
| **Spatial Generalization (LOSO)** | Single-Station Overfitting Risk | $\mathbf{46.33 \pm 1.65\ \mu\text{g}/\text{m}^3}$ (12-Fold Mean) | Stable domain transfer across 411,165 samples | `experiment7_loso_generalisation.json` |
| **Same-Model Numerical Drift** | Bit-level identity expected | $\mathbf{0.000000\times 10^0}$ (Max absolute drift) | Zero numerical drift between Pandas and Buffer | Live test evaluation ($N=42,554$) |
| **Regression Test Suite** | 52 Total Unit Tests | **52 Passed, 0 Failed ($100.0\%$)** | Verified in 27.65s execution | `pytest tests/ -v` |

---

## 2. Task 1 & 2: Detailed Verification of Performance Claims & Inconsistencies

### 2.1 Artifact Storage Footprint Recalculation

An exact file-level disk audit of `artifacts/models/` was performed using `os.stat()` on Windows NTFS. The directory houses **29 serialized artifacts** totaling **118,704,192 bytes (113.2051 MB)**.

```
Artifact Name                         Exact Bytes      Disk Size (MB)   Model Role
------------------------------------------------------------------------------------------------------
conformal_calibrator_h1.joblib          1,231,379           1.1743 MB   1h Normalized Conformal Engine
conformal_calibrator_h6.joblib          1,314,291           1.2534 MB   6h Normalized Conformal Engine
conformal_calibrator_h24.joblib           935,099           0.8918 MB   24h Normalized Conformal Engine
conformal_calibrator_std_h1.joblib      1,231,379           1.1743 MB   1h Standard Conformal Engine
conformal_calibrator_std_h6.joblib      1,314,291           1.2534 MB   6h Standard Conformal Engine
conformal_calibrator_std_h24.joblib       935,099           0.8918 MB   24h Standard Conformal Engine
hist_gradient_boosting_h1.joblib          609,648           0.5814 MB   1h HistGBDT Baseline
hist_gradient_boosting_h6.joblib          609,632           0.5814 MB   6h HistGBDT Baseline
hist_gradient_boosting_h24.joblib         609,536           0.5813 MB   24h HistGBDT Baseline
linear_regression_h1.joblib                 3,596           0.0034 MB   1h OLS Linear Baseline
linear_regression_h6.joblib                 3,596           0.0034 MB   6h OLS Linear Baseline
linear_regression_h24.joblib                3,596           0.0034 MB   24h OLS Linear Baseline
lstm_h1.joblib                            263,865           0.2516 MB   1h PyTorch LSTM (2-layer, 64-dim)
lstm_h6.joblib                            263,556           0.2513 MB   6h PyTorch LSTM (2-layer, 64-dim)
lstm_h24.joblib                           263,476           0.2513 MB   24h PyTorch LSTM (2-layer, 64-dim)
persistence_h1.joblib                       1,283           0.0012 MB   1h Persistence Baseline
persistence_h6.joblib                       1,283           0.0012 MB   6h Persistence Baseline
persistence_h24.joblib                      1,283           0.0012 MB   24h Persistence Baseline
random_forest_h1.joblib                25,781,411          24.5871 MB   1h Random Forest (60 trees, d=14)
random_forest_h6.joblib                35,882,291          34.2200 MB   6h Random Forest (60 trees, d=14)
random_forest_h24.joblib               43,895,027          41.8616 MB   24h Random Forest (60 trees, d=14)
ridge_h1.joblib                             3,212           0.0031 MB   1h Ridge L2 Baseline (alpha=10)
ridge_h6.joblib                             3,212           0.0031 MB   6h Ridge L2 Baseline (alpha=10)
ridge_h24.joblib                            3,212           0.0031 MB   24h Ridge L2 Baseline (alpha=10)
xgboost_h1.joblib                       1,213,904           1.1577 MB   1h XGBoost Baseline (180 trees, d=7)
xgboost_h6.joblib                       1,130,027           1.0777 MB   6h XGBoost Baseline (180 trees, d=7)
xgboost_h24.joblib                        600,347           0.5725 MB   24h XGBoost Baseline (180 trees, d=7)
xgboost_pruned_h6.joblib                  453,852           0.4328 MB   6h XGBoost Pruned (90 trees, d=6)
xgboost_fast_h24.joblib                   141,809           0.1352 MB   24h XGBoost Fast (50 trees, d=5)
------------------------------------------------------------------------------------------------------
TOTAL PHYSICAL REPOSITORY SIZE:       118,704,192 bytes   113.2051 MB   (29 Serialized Artifacts)
```

#### Reconciled Suite Reductions
1. **Recommended Multi-Horizon Pareto Deployment Suite:**
   $$\text{Pareto Suite} = \text{xgboost\_h1 } (1.1577\text{ MB}) + \text{xgboost\_pruned\_h6 } (0.4328\text{ MB}) + \text{xgboost\_fast\_h24 } (0.1352\text{ MB}) = \mathbf{1.7257\text{ MB}}$$
2. **Baseline Random Forest Deployment Suite:**
   $$\text{RF Suite} = 24.5871 + 34.2200 + 41.8616 = \mathbf{100.6686\text{ MB}} \implies \mathbf{98.29\% \text{ reduction } (58.33\times \text{ smaller})}$$
3. **Baseline XGBoost Deployment Suite:**
   $$\text{XGB Suite} = 1.1577 + 1.0777 + 0.5725 = \mathbf{2.8079\text{ MB}} \implies \mathbf{38.54\% \text{ reduction } (1.63\times \text{ smaller})}$$
4. **Footprint Reduction vs Complete Benchmark Archive:**
   $$\text{Archive Footprint Reduction} = \frac{113.2051 - 1.7257}{113.2051} = \mathbf{98.48\% \text{ reduction } (65.60\times \text{ smaller})}$$

---

### 2.2 RAM and Memory Reduction Recalculation

Verified directly from `artifacts/metrics/memory_optimization.json`:
1. **Active DataFrame Memory Downcasting:**
   - 420,768 rows $\times$ 70 columns (engineered feature cache).
   - Standard 64-bit IEEE float: **$227.38\text{ MB}$**.
   - Downcasted 32-bit IEEE float (`np.float32`): **$121.45\text{ MB}$**.
   - Net RAM Savings: **$46.59\%$ ($105.93\text{ MB}$ liberated, $1.87\times$ smaller)**.
2. **In-Process Memory Leak Audit:**
   - Continuous 10,000-request evaluation loop (`memory_leak_verification`):
   - Initial Process RAM: $347.90\text{ MB}$ $\to$ Final Process RAM: $350.14\text{ MB}$.
   - Net growth after garbage collection: $+2.24\text{ MB}$ ($< 0.0003\text{ MB}$ per request), verifying memory stability.

---

### 2.3 API Latency Reconciliation & Performance Budget Audit

A detailed latency audit was performed comparing the **direct C++ NumPy buffer engine** against the **end-to-end REST API service layer** (`AirCastInferenceService`):

#### Steady-State Latency Profile (200 requests per horizon)

```
Horizon   Model Artifact               Mean Latency    Median Latency    P95 Latency    P99 Latency    Cold Warmup
------------------------------------------------------------------------------------------------------------------
+1h       xgboost_h1.joblib (180 t)       28.05 ms        35.98 ms         44.13 ms       57.86 ms        73.91 ms
+6h       xgboost_pruned_h6.joblib (90 t) 35.04 ms        38.00 ms         43.36 ms       66.70 ms       149.20 ms
+24h      xgboost_fast_h24.joblib (50 t)  24.72 ms        13.84 ms         43.32 ms       48.43 ms       143.29 ms
```

#### Latency Component Breakdown
- **Feature Parsing & Pandas DataFrame Conversion:** $3.16\text{–}4.45\text{ ms}$ ($12.6\%\text{–}16.5\%$).
- **Single-Row XGBoost C++ DMatrix Prediction:** $19.74\text{–}30.56\text{ ms}$ ($84.3\%\text{–}87.3\%$).
- **Conformal Margin Calculation:** $< 0.02\text{ ms}$ ($< 0.1\%$).

#### Performance Budget Reconciliation
- **Direct C++ NumPy Array Inference:** **$4.33\text{ ms}$** (XGBoost), **$0.15\text{ ms}$** (Ridge), **$1.49\text{ ms}$** (LSTM). **Meets the $\le 15.0\text{ ms}$ budget.**
- **End-to-End REST Single-Record API:** **P95 latency is $43.3\text{–}44.1\text{ ms}$**, failing the $15.0\text{ ms}$ P95 budget due to single-row Pandas DataFrame instantiation and Python-to-C++ DMatrix creation overhead.
- **Cold Warmup Calls:** Range from $73.9\text{ ms}$ to $149.2\text{ ms}$ due to lazy disk unpickling and OpenMP thread pool allocation.

---

### 2.4 Batch Serving Throughput Recalculation

Verified directly from `artifacts/metrics/inference_optimization.json` (`batch_scaling_benchmarks`):

```
Batch Size (b)      XGBoost Latency (ms)    XGBoost (samples/s)     Ridge (samples/s)      LSTM (samples/s)
---------------------------------------------------------------------------------------------------------
b = 1                          3.489 ms               286.6 s/s            7,946.3 s/s            322.6 s/s
b = 16                         3.983 ms             4,017.1 s/s          124,031.0 s/s          4,266.7 s/s
b = 64                         4.180 ms            15,311.0 s/s          460,431.7 s/s         14,349.8 s/s
b = 256                        4.757 ms            53,815.4 s/s        1,684,210.5 s/s         31,604.9 s/s
b = 512                        5.281 ms            96,951.3 s/s        2,860,335.2 s/s         37,647.1 s/s
b = 1024                       5.427 ms           188,688.3 s/s        4,335,492.7 s/s         41,798.5 s/s
```

Batch serving amortizes Python/C++ boundary overhead, driving throughput to **$188,688.3\text{ samples/sec}$** for XGBoost and **$4,335,492.7\text{ samples/sec}$** for Ridge Regression at $b=1024$.

---

## 3. Conformal Coverage Recalculation & Limitations

Empirical coverage was recalculated across all test split records ($N=42,554$ per horizon) using `conformal_calibrator_h{1,6,24}.joblib`:

```
Metric                 Horizon +1h (Inertia)        Horizon +6h (Mesoscale)       Horizon +24h (Synoptic)
---------------------------------------------------------------------------------------------------------
Calibrator Artifact    conformal_calibrator_h1      conformal_calibrator_h6       conformal_calibrator_h24
Conformal Method       Normalized Adaptive          Normalized Adaptive           Normalized Adaptive
Nominal Coverage       90.0%                        90.0%                         90.0%
Empirical Coverage     89.07% (-0.93% gap)          85.26% (-4.74% gap)           79.81% (-10.19% gap)
Mean Interval Width    44.88 ug/m3                  131.29 ug/m3                  181.15 ug/m3
Median Interval Width  35.54 ug/m3                  111.77 ug/m3                  174.62 ug/m3
Good (0-35 ug/m3)      92.53% (width 20.82)         82.93% (width 60.71)          79.97% (width 158.96)
Moderate (35-75)       85.80% (width 33.17)         89.25% (width 100.58)         98.98% (width 147.14)
Unhealthy Sens. (75-115)87.29% (width 44.10)        91.81% (width 131.34)         92.82% (width 165.57)
Unhealthy (115-150)    86.83% (width 54.72)         88.91% (width 163.03)         87.48% (width 193.29)
Very Unhealthy(150-250)88.61% (width 72.18)         87.87% (width 219.99)         71.95% (width 228.81)
Hazardous (> 250)      86.34% (width 113.39)        69.40% (width 304.46)         28.05% (width 271.82)
Pooled Test Coverage   84.72% (across all 127,326 test samples pooled over 3 horizons)
```

### Clarification on the 91.4% Coverage Claim
- The historic $91.4\%$ claim was derived in Phase 7 (`research/uncertainty_and_explainability.md`) using a static split conformal predictor with a single fixed margin ($56.20\ \mu\text{g}/\text{m}^3$) on 42,278 test samples ($38,642$ covered).
- When evaluating horizon-specific normalized conformal calibrators on the chronological test split, empirical coverage is **$89.07\%$ at $+1\text{h}$**, **$85.26\%$ at $+6\text{h}$**, and **$79.81\%$ at $+24\text{h}$** (pooled: **$84.72\%$**).
- **Physical Reason for Long-Horizon Degradation:** Time-series non-stationarity and extreme winter stagnation episodes violate exchangeability between calibration (Jan–Sep 2016) and test (Oct 2016–Feb 2017) partitions, causing coverage on extreme events ($> 250\ \mu\text{g}/\text{m}^3$) to drop to $28.05\%$ at $+24\text{h}$.

---

## 4. Numerical Equivalence Verification

Evaluated across all 42,554 test records:

1. **Same Model Comparison (Pandas DataFrame vs. Preallocated NumPy Buffer):**
   - **Horizon +1h (`xgboost_h1.joblib`):** $\max |\hat{y}_{\text{df}} - \hat{y}_{\text{buf}}| = \mathbf{0.000000\times 10^0}$.
   - **Horizon +6h (`xgboost_h6.joblib`):** $\max |\hat{y}_{\text{df}} - \hat{y}_{\text{buf}}| = \mathbf{0.000000\times 10^0}$.
   - **Horizon +24h (`xgboost_h24.joblib`):** $\max |\hat{y}_{\text{df}} - \hat{y}_{\text{buf}}| = \mathbf{0.000000\times 10^0}$.
   - **Conclusion:** Exact bit-level numerical equivalence is verified when evaluating the same model artifact.

2. **Cross-Model Comparison (Baseline vs. Recommended Pareto Suite):**
   - **Horizon +1h:** Same artifact (`xgboost_h1.joblib`), $\Delta\text{MAE} = 0.0000$.
   - **Horizon +6h:** Baseline (180 trees) vs Pruned (90 trees) $\implies$ Max diff = $89.46\ \mu\text{g}/\text{m}^3$, Mean diff = $4.94\ \mu\text{g}/\text{m}^3$, $\Delta\text{MAE} = +0.0809\ \mu\text{g}/\text{m}^3$.
   - **Horizon +24h:** Baseline (180 trees) vs Fast (50 trees) $\implies$ Max diff = $103.47\ \mu\text{g}/\text{m}^3$, Mean diff = $9.11\ \mu\text{g}/\text{m}^3$, $\Delta\text{MAE} = -1.1817\ \mu\text{g}/\text{m}^3$ ($1.76\%$ accuracy gain).
   - **Conclusion:** The pruned and fast models are structurally distinct tree ensembles, NOT numerically identical to baselines.

---

## 5. Spatial Generalization Audit (12-Fold LOSO)

Verified from `artifacts/metrics/experiment7_loso_generalisation.json` (411,165 total evaluated samples across 12 folds):

```
Fold   Held-out Station       Domain Role           Test Samples    Test MAE    Test RMSE    Test R2    Training Stations
-------------------------------------------------------------------------------------------------------------------------
 1     Aotizhongxin           Urban Core                   34,067     45.8528      63.6086     0.4009    11 stations
 2     Changping              Suburban                     34,218     43.8345      59.9038     0.3148    11 stations
 3     Dingling               Regional Background          34,213     44.1467      60.0775     0.3098    11 stations
 4     Dongsi                 Urban Core                   34,242     48.0307      67.3196     0.3960    11 stations
 5     Guanyuan               Urban Core                   34,376     46.2431      63.8003     0.3791    11 stations
 6     Gucheng                Industrial Corridor          34,346     46.9723      67.2358     0.3413    11 stations
 7     Huairou                Suburban                     34,039     43.4518      59.3024     0.3075    11 stations
 8     Nongzhanguan           Urban Core                   34,364     47.8548      67.0957     0.3951    11 stations
 9     Shunyi                 Suburban                     34,079     47.4498      66.5596     0.3293    11 stations
10     Tiantan                Urban Core                   34,315     46.2658      64.2603     0.3701    11 stations
11     Wanliu                 Urban Core                   34,610     47.1013      65.3387     0.3642    11 stations
12     Wanshouxigong          Urban Core                   34,296     48.7184      69.2365     0.3522    11 stations
-------------------------------------------------------------------------------------------------------------------------
12-FOLD MACRO-AVERAGED:                                   411,165     46.33 +- 1.65   64.48      0.3550
```

### Methodological Context
In each fold, 11 stations train the model, while the 12th station is held out. **Crucially**, during inference on the held-out station, the model receives that station's **own historical local telemetry** ($\tau \le t$), including local autoregressive lags and meteorological measurements. This verifies **spatial domain transfer to a newly instrumented monitoring station**, NOT prediction at an unmonitored geographic location.

---

## 6. Final Test Suite & Release Verification

1. **Automated Pytest Suite Execution:**
   - Command: `pytest tests/ -v`
   - Result: **52 passed, 6 warnings in 27.65s (100.0% pass rate)**.
2. **Artifact Integrity:**
   - All 29 model and calibrator binaries in `artifacts/models/` verified loadable and intact.
3. **Inference Service Alias Routing:**
   - Querying `model_name="recommended"` in `AirCastInferenceService` correctly routes to `xgboost_h1.joblib` (+1h), `xgboost_pruned_h6.joblib` (+6h), and `xgboost_fast_h24.joblib` (+24h) with active calibrated conformal bounds.

---

## 7. Final Sign-Off & Portfolio Release Status

Phase 11 has been rigorously reconciled, all empirical discrepancies resolved, and all unsupported claims removed.

**RELEASE DECISION: APPROVED FOR RESEARCH PORTFOLIO RELEASE (v1.0.0-research)**
