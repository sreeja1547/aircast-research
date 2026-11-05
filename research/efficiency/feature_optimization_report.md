# AirCast Research: Stage 2 — Feature Efficiency & Selection Report

**Phase 11: Computational Efficiency, Model Optimization & Inference Acceleration**  
**Author:** Principal ML Systems & Performance Engineer  
**Timestamp:** 2026-11-05T20:34:00+05:30  
**Artifacts:**  
- [`artifacts/metrics/feature_efficiency_comparison.json`](../../artifacts/metrics/feature_efficiency_comparison.json)  
- [`artifacts/plots/feature_efficiency_comparison.png`](../../artifacts/plots/feature_efficiency_comparison.png)

---

## 1. Executive Summary

This report documents the systematic evaluation of feature groups, correlation/collinearity diagnostics, and controlled validation-based feature pruning. The objective is to identify the **minimum useful feature subset** per forecasting horizon that preserves predictive fidelity while drastically reducing memory bandwidth, training duration, and inference latency.

### Key Optimization Highlights:
- **Horizon $h=1\text{h}$:** Feature count reduced from **47 to 16 features ($-66.0\%$ reduction)**. Training duration accelerated by **$2.18\times$**, inference throughput increased by **$+56.0\%$**, and test matrix RAM decreased by **$-63.7\%$** with only $+0.218\ \mu\text{g}/\text{m}^3$ test MAE delta.
- **Horizon $h=6\text{h}$:** Feature count reduced from **47 to 24 features ($-48.9\%$ reduction)**. RAM consumption reduced by **$-47.3\%$**, throughput increased by **$+42.1\%$**, and MAE delta was $+0.209\ \mu\text{g}/\text{m}^3$.
- **Horizon $h=24\text{h}$:** Feature count reduced from **47 to 24 features ($-48.9\%$ reduction)**. RAM consumption reduced by **$-47.3\%$**, with MAE delta of $+0.403\ \mu\text{g}/\text{m}^3$.

---

## 2. Methodology & Experimental Controls

1. **Zero Test Data Snooping:** All feature rankings, collinearity thresholds ($|r| > 0.92$), and candidate subset selections ($K \in \{10, 16, 24, 32\}$) were computed strictly on the **2016 Validation partition** (January 2 to September 30, 2016; 77,316 samples).
2. **Untouched Test Evaluation:** Only after selecting candidate subsets was a single final evaluation run executed on the chronologically purged **Test partition** (October 2016 to February 2017; ~42,278 samples).
3. **Collinearity Pruning:** Over 31 highly collinear pairs ($|r| > 0.92$) were identified in the 47-feature matrix (e.g., `PM2.5_roll_mean_6` vs. `PM2.5_roll_mean_12`, and `PM2.5` vs. `PM2.5_lag_1`). Redundant rolling statistics evaluating near-identical temporal windows were safely eliminated.

---

## 3. Horizon-Specific Feature Optimization Breakdown

### Table 1: End-to-End Performance Impact on Untouched Test Set

| Horizon ($h$) | Feature Config | Feature Count | Test MAE ($\mu\text{g}/\text{m}^3$) | Test RMSE ($\mu\text{g}/\text{m}^3$) | Test $R^2$ | Training Time (s) | Throughput (s/s) | Test Matrix RAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h=1\text{h}$** | Baseline (`combined_all`) | 47 | $11.607$ | $23.443$ | $0.9494$ | $14.85\text{ s}$ | $55,010$ | $14.77\text{ MB}$ |
| **$h=1\text{h}$** | **Optimized Compact** | **16** | **$11.825$** | **$23.731$** | **$0.9481$** | **$6.83\text{ s}$** | **$85,793$** | **$5.36\text{ MB}$** |
| | *Change / Gain* | **$-66.0\%$** | *$+0.218$* | *$+0.288$* | *$-0.0013$* | **$2.18\times$ Faster** | **$+56.0\%$** | **$-63.7\%$ RAM** |
| **$h=6\text{h}$** | Baseline (`combined_all`) | 47 | $36.790$ | $53.864$ | $0.7326$ | $14.20\text{ s}$ | $52,100$ | $14.75\text{ MB}$ |
| **$h=6\text{h}$** | **Optimized Compact** | **24** | **$36.999$** | **$54.210$** | **$0.7292$** | **$7.85\text{ s}$** | **$74,050$** | **$7.77\text{ MB}$** |
| | *Change / Gain* | **$-48.9\%$** | *$+0.209$* | *$+0.346$* | *$-0.0034$* | **$1.81\times$ Faster** | **$+42.1\%$** | **$-47.3\%$ RAM** |
| **$h=24\text{h}$** | Baseline (`combined_all`) | 47 | $67.086$ | $93.181$ | $0.2018$ | $13.62\text{ s}$ | $48,200$ | $14.67\text{ MB}$ |
| **$h=24\text{h}$** | **Optimized Compact** | **24** | **$67.489$** | **$93.580$** | **$0.1949$** | **$7.19\text{ s}$** | **$66,780$** | **$7.73\text{ MB}$** |
| | *Change / Gain* | **$-48.9\%$** | *$+0.403$* | *$+0.399$* | *$-0.0069$* | **$1.89\times$ Faster** | **$+38.5\%$** | **$-47.3\%$ RAM** |

---

## 4. Retained Physical Feature Subsets

### 1-Hour Horizon (16 Features):
At $h=1\text{h}$, immediate particulate inertia and rapid boundary turbulence dominate:
1. `PM2.5` (Instantaneous concentration)
2. `PM2.5_roll_min_3` (Short-window baseline)
3. `PM2.5_roll_mean_3` (Short-window trajectory)
4. `PM10` (Coarse fraction ratio)
5. `PM2.5_roll_max_3` (Short-window peak)
6. `WSPM` (Wind speed ventilation)
7. `wind_v` (North-South meridional advection)
8. `dew_point_depression` (Atmospheric moisture deficit)
9. `RAIN` (Precipitation scavenging)
10. `PM2.5_roll_max_6` (6-hour peak)
11. `PM2.5_roll_std_3` (Immediate volatility)
12. `has_rain` (Binary wet deposition flag)
13. `PM2.5_lag_1` (Immediate autoregressive inertia)
14. `PM2.5_roll_max_12` (Diurnal peak indicator)
15. `CO` (Incomplete combustion tracer)
16. `PM2.5_roll_max_24` (Synoptic episode threshold)

### 6-Hour and 24-Hour Horizons (24 Features):
At longer horizons, mesoscale advection, diurnal cycling, and trace gas precursors become essential. The selected 24-feature set retains the 16 features above plus:
17. `PM2.5_roll_mean_24` (Synoptic accumulation mean)
18. `PM2.5_lag_3` (Multi-hour inertia)
19. `NO2` (Photochemical precursor)
20. `PM2.5_roll_std_12` (Half-day boundary-layer dispersion)
21. `wd_cos` (Wind orientation harmonic)
22. `hour_cos` (Diurnal solar heating cycle)
23. `PM2.5_lag_48` (Synoptic recurrence lag)
24. `PM2.5_roll_mean_6` (6-hour meso-scale mean)

---

## 5. Engineering Conclusion & Deployment Recommendation

Pruning 47 features down to 16–24 features delivers:
1. **$1.8\times\text{ to }2.2\times$ faster training**, accelerating iterative retraining cycles.
2. **$+38\%\text{ to }+56\%$ higher inference throughput**, allowing higher queries per second on single-node deployments.
3. **$\sim 50\text{--}64\%$ lower matrix RAM consumption**, reducing process memory footprint from 203 MB down to ~75–100 MB.
4. Minimal accuracy trade-off ($<0.4\ \mu\text{g}/\text{m}^3$ MAE change across all lead times).
