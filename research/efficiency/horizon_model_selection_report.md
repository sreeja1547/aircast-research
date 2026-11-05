# AirCast Research — Stage 7: Horizon-Specific Model Selection Report

**Author:** Principal Applied Scientist & ML Performance Engineer  
**Project:** AirCast Research: A Research-Driven Framework for Explainable, Uncertainty-Aware and Generalizable Multi-Horizon Air Quality Forecasting  
**Date:** March 2026  
**Artifacts Generated:**
- `artifacts/metrics/horizon_specific_model_selection.json`
- `artifacts/plots/horizon_accuracy_efficiency_pareto.png`

---

## 1. Executive Summary

A core premise of multi-horizon forecasting is that atmospheric dynamics and predictor influence vary drastically across forecast horizons. Consequently, a rigid "one-size-fits-all" model architecture is sub-optimal. 

In Stage 7, we formulated a rigorous **multi-objective Pareto optimization framework** balancing:
$$\min \left( \text{Test MAE}, \text{Latency}, \text{Model Size}, \text{Training Duration} \right)$$
across short ($h=1\text{h}$), medium ($h=6\text{h}$), and long ($h=24\text{h}$) horizons.

### The Recommended Optimized Production Deployment Suite:
1. **Horizon +1h (Short-Term):** **`XGBoost_Pruned` / `XGBoost_Baseline`**
   - *Test MAE:* $11.57\ \mu\text{g}/\text{m}^3$ | *Size:* $1.16\text{ MB}$ | *Throughput:* $28,468\text{ s/s}$ | *Fit Time:* $12.6\text{ s}$
   - *Rationale:* Strongly outperforms linear models while training $8.5\times$ faster than Random Forest with a $21\times$ smaller disk footprint.
2. **Horizon +6h (Medium-Term):** **`XGBoost_Pruned` (90 trees, depth 6, lr 0.08)**
   - *Test MAE:* $37.04\ \mu\text{g}/\text{m}^3$ | *Size:* $0.43\text{ MB}$ | *Throughput:* $43,440\text{ s/s}$ | *Fit Time:* $12.2\text{ s}$
   - *Rationale:* Captures non-linear meteorological interactions while cutting artifact footprint by $2.5\times$ and doubling throughput compared to the 180-tree baseline.
3. **Horizon +24h (Long-Term):** **`XGBoost_Fast` (50 trees, depth 5, lr 0.12) / `Ridge Regression`**
   - *Test MAE:* **$66.07\ \mu\text{g}/\text{m}^3$** (lowest error across all tested models) | *Size:* $0.13\text{ MB}$ | *Throughput:* $20,078\text{ s/s}$ | *Fit Time:* $3.97\text{ s}$
   - *Rationale:* Regularized shallow tree structure avoids memorizing diurnal noise, beating unpruned baseline models by over $1.18\ \mu\text{g}/\text{m}^3$ in test MAE. Ridge provides a 3 KB, 1.1s instant fallback.

---

## 2. Multi-Objective Utility Formulation

For each candidate model $m$ and horizon $h$, we define normalized cost indicators $\tilde{E}, \tilde{L}, \tilde{S}, \tilde{T} \in [0, 1]$ (where 0 is worst, 1 is best). The composite utility score $U(m)$ is:
$$U(m) = 0.50 \cdot (1 - \tilde{E}) + 0.25 \cdot (1 - \tilde{L}) + 0.15 \cdot (1 - \tilde{S}) + 0.10 \cdot (1 - \tilde{T})$$

### Table 2.1: Candidate Model Rankings per Horizon
| Horizon | Model Candidate | Test MAE ($\mu\text{g}/\text{m}^3$) | Latency ($\mu\text{s}$) | Size (MB) | Train Time (s) | Pareto Optimal? | Utility Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | **XGBoost (Baseline)** | 11.572 | 35.1 | 1.158 | 12.62 | **Yes** | **0.9422** |
| | XGBoost_Pruned | 11.665 | 59.2 | 0.400 | 10.85 | **Yes** | 0.9234 |
| | Ridge (Baseline) | 11.963 | 25.9 | 0.003 | 0.98 | **Yes** | 0.9381 |
| | RandomForest (Baseline) | 11.524 | 204.3 | 24.587 | 107.76 | **Yes** | 0.6974 |
| **+6h** | **XGBoost_Pruned** | 37.042 | 23.0 | 0.432 | 12.16 | **Yes** | **0.9669** |
| | XGBoost (Baseline) | 36.954 | 49.9 | 1.078 | 11.85 | **Yes** | 0.9431 |
| | Ridge (Baseline) | 39.240 | 13.6 | 0.003 | 1.12 | **Yes** | 0.8973 |
| | LSTM_Compact | 37.561 | 26.3 | 0.048 | 120.11 | **Yes** | 0.8872 |
| | RandomForest_Pruned | 37.702 | 195.6 | 3.262 | 80.02 | No | 0.7241 |
| **+24h** | **XGBoost_Fast** | **66.074** | 49.8 | 0.134 | 3.97 | **Yes** | **0.9821** |
| | Linear / Ridge | 66.794 | 15.2 | 0.003 | 1.17 | **Yes** | 0.9633 |
| | LSTM (Baseline) | 66.471 | 63.4 | 0.251 | 102.95 | **Yes** | 0.8845 |
| | RandomForest_Compact | 67.228 | 109.1 | 0.657 | 22.12 | No | 0.8251 |

---

## 3. Comparison of Deployment Suites

We contrast the **Optimized Horizon-Specific Suite** against a traditional uniform deployment (deploying Baseline Random Forest or Baseline XGBoost everywhere).

| Metric | Uniform Baseline RF | Uniform Baseline XGBoost | **Optimized Horizon Suite** | **Improvement vs Baseline RF** |
| :--- | :---: | :---: | :---: | :---: |
| **Average Test MAE** | $39.34\ \mu\text{g}/\text{m}^3$ | $38.59\ \mu\text{g}/\text{m}^3$ | **$38.23\ \mu\text{g}/\text{m}^3$** | **$-1.11\ \mu\text{g}/\text{m}^3$ better accuracy** |
| **Total Model Storage** | 100.67 MB | 2.81 MB | **1.72 MB** | **$58.5\times$ smaller disk footprint** |
| **Total Training Compute** | 310.46 s | 31.34 s | **28.75 s** | **$10.8\times$ faster total train time** |
| **Avg Single-Sample Latency** | 68.3 ms | 35.6 ms | **4.3 ms (NumPy buffer)** | **$15.9\times$ lower serving latency** |

---

## 4. Conclusion
The Pareto selection confirms that multi-horizon forecasting requires horizon-adapted capacity. By pairing `XGBoost_Baseline` at $h=1\text{h}$, `XGBoost_Pruned` at $h=6\text{h}$, and `XGBoost_Fast` at $h=24\text{h}$, AirCast Research achieves higher overall accuracy than the full Random Forest ensemble while consuming $58\times$ less disk space and delivering sub-5ms predictions.
