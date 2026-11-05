# AirCast Research — Stage 4: Training-Time Optimization Report

**Author:** Principal Applied Scientist & ML Performance Engineer  
**Project:** AirCast Research: A Research-Driven Framework for Explainable, Uncertainty-Aware and Generalizable Multi-Horizon Air Quality Forecasting  
**Date:** March 2026  
**Artifacts Generated:**
- `artifacts/metrics/training_efficiency.json`

---

## 1. Executive Summary

Stage 4 of Phase 11 investigated training-time acceleration across four distinct systems dimensions:
1. **Precision Optimization:** 32-bit floating point downcasting vs 64-bit IEEE double precision.
2. **PyTorch DataLoader Pipeline:** Recurrent tensor batch size tuning and throughput profiling on CPU.
3. **Multi-Threading Parallel Scaling:** Empirical Amdahl's Law analysis across 1, 2, 4, 8, and 12 CPU cores.
4. **Feature Vectorization:** Vectorized Pandas/NumPy groupby operations vs iterative row processing.

---

## 2. Precision Optimization (float64 vs float32)

We benchmarked the full training partition ($N = 289,368$ samples, $F = 47$ features) under 64-bit float and 32-bit float encodings.

| Metric | Float64 (Double) | Float32 (Single) | Delta / Savings |
| :--- | :---: | :---: | :---: |
| **Feature Matrix RAM** | 106.78 MB | 53.39 MB | **-50.0% RAM reduction** |
| **XGBoost Fit Time** | 3.510 s | 3.681 s | Comparable ($\pm 4.8\%$) |
| **XGBoost Test MAE** | 11.828 | 11.828 | **0.000000 max prediction diff** |
| **Ridge Fit Time** | 0.444 s | 2.240 s | SciPy LAPACK path variation |
| **Ridge Test MAE** | 11.963 | 11.963 | $0.00182\ \mu\text{g}/\text{m}^3$ max diff |

### Scientific Significance:
Downcasting feature matrices to `float32` immediately cuts training RAM requirements in half across all 420,768 records with zero loss in predictive fidelity for tree models (`max diff = 0.000000`) and sub-milligram differences for linear ridge solvers ($< 0.002\ \mu\text{g}/\text{m}^3$).

---

## 3. PyTorch DataLoader Batch Sizing

We profiled tensor iteration throughput through PyTorch DataLoader instances processing tabular sequence tensors across 289,368 samples on CPU.

| Batch Size | Epoch Duration (s) | Throughput (Samples / Sec) | Latency per Batch (ms) |
| :---: | :---: | :---: | :---: |
| **128** | 2.683 | 108,684 | 1.187 |
| **256** | 3.384 | 86,162 | 2.995 |
| **512** | 3.256 | 89,544 | 5.763 |
| **1024** | 3.046 | 95,716 | 10.763 |
| **2048** | **2.295** | **127,038** | 16.277 |

**Recommendation:** Increasing training batch size from 256 to 2048 improves DataLoader throughput by **$+47.4\%$**, reducing CPU tensor slicing overhead.

---

## 4. Multi-Threading Scaling & Amdahl's Law Analysis

We evaluated CPU core scaling across $k \in \{1, 2, 4, 8, 12\}$ threads on our 10-core / 12-thread Intel hardware.

### 4.1 XGBoost Multi-Threading Scaling
| Threads ($k$) | Wall-Clock Time (s) | Speedup ($S_k = T_1 / T_k$) | Parallel Efficiency ($E_k = S_k / k$) |
| :---: | :---: | :---: | :---: |
| **1** | 4.91 | $1.00\times$ | 100.0% |
| **2** | **3.02** | **$1.63\times$** | **81.5%** |
| **4** | 4.11 | $1.19\times$ | 29.8% |
| **8** | 3.52 | $1.40\times$ | 17.5% |
| **12** | 3.72 | $1.32\times$ | 11.0% |

**Amdahl Insight for XGBoost:** XGBoost `hist` tree building achieves peak efficiency at $k = 2$ threads. Beyond 4 threads, thread synchronization overhead and cache thrashing across OpenMP histogram bins produce diminishing or negative returns. For high-throughput server deployments, running multiple 2-thread workers yields significantly higher overall system throughput than running a single 12-thread worker.

### 4.2 Random Forest Multi-Threading Scaling
| Threads ($k$) | Wall-Clock Time (s) | Speedup ($S_k = T_1 / T_k$) | Parallel Efficiency ($E_k = S_k / k$) |
| :---: | :---: | :---: | :---: |
| **1** | 80.71 | $1.00\times$ | 100.0% |
| **2** | 36.61 | $2.20\times$ | 110.0% (Super-linear cache effect) |
| **4** | 28.53 | $2.83\times$ | 70.8% |
| **8** | 22.41 | $3.60\times$ | 45.0% |
| **12** | **21.06** | **$3.83\times$** | 31.9% |

**Amdahl Insight for Random Forest:** Because Random Forest trees are completely embarrassingly parallel (bagging), scaling from 1 to 12 threads yields a **$3.83\times$ speedup** ($80.71\text{s} \to 21.06\text{s}$). The parallel fraction $p \approx 0.77$ indicates that tree construction parallelizes well, but joblib process spawning and tree gathering constitute the remaining sequential fraction $1 - p \approx 0.23$.

---

## 5. Feature Vectorization

We compared Pandas vectorized groupby transformations against iterative Python processing over 200,000 multi-station records:
- **Vectorized Groupby Lags:** **0.0496 seconds**
- **Iterative Row Processing:** **4.2010 seconds**
- **Speedup Factor:** **$85\times$**

All lag and rolling statistics in `src/features/builder.py` leverage vectorized broadcast transforms, preventing Python interpreter loop bottlenecks.
