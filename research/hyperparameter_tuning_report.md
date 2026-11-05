# AirCast Research — Experiment 4: Hyperparameter Optimization and Generalization Analysis

**Project:** AirCast Research: A Research-Driven Framework for Explainable, Uncertainty-Aware and Generalizable Multi-Horizon Air Quality Forecasting  
**Author:** Senior Applied Scientist & Machine Learning Architect  
**Validation Protocol:** Chronological Train/Val/Test with 24-Hour Purge Embargo  
**Model Architecture:** XGBoost Regressor (Hist Gradient Boosting)  
**Target Variable:** $PM_{2.5}(t+24)$ (24-Hour Ahead Lead Time)  
**Dataset Partitions:** Training ($\le 2015\text{-}12\text{-}31$, 291,574 samples), Validation ($2016\text{-}01\text{-}02\text{ to }2016\text{-}09\text{-}30$, 77,316 samples), Test ($2016\text{-}10\text{-}02\text{ to }2017\text{-}02\text{-}28$, 42,278 samples)  

---

## 1. Executive Summary & Experimental Design

Long-horizon atmospheric forecasting ($h=24\text{h}$) is inherently ill-posed due to non-linear meteorological dispersion and chaotic synoptic transitions. In such noisy regimes, overly expressive tree architectures risk memorizing transient atmospheric eddies, leading to acute generalization degradation on unseen future seasons.

Experiment 4 conducts a systematic hyperparameter optimization study on the **validation partition** to identify the Pareto-optimal configuration balancing bias, variance, and computational efficiency for 24-hour forecasting.

### Search Space & Optimization Protocol
We explore an orthogonal grid of 18 configurations across three key architectural dimensions:
- **Shrinkage Factor / Learning Rate ($\eta \in \{0.03, 0.06, 0.10\}$):** Controls step-size regularization along the negative gradient.
- **Maximum Tree Depth ($d \in \{5, 7, 9\}$):** Dictates maximum interaction order and functional hypothesis complexity.
- **Stochastic Subsample Ratio ($s \in \{0.75, 0.90\}$):** Fraction of training instances sampled per boosting iteration to inject bagging variance reduction.
- **Strict Validation Quarantine:** Model selection is conducted **strictly on the validation partition** ($R^2$ and RMSE optimization). The held-out test split is evaluated only once on the winning model to ensure unbiased scientific reporting.

---

## 2. Validation Response Surface Results

Below is the complete response surface across all 18 evaluated configurations on the validation split:

| Config ID | Learning Rate ($\eta$) | Max Depth ($d$) | Subsample ($s$) | Val MAE ($\mu\text{g}/\text{m}^3$) | Val RMSE ($\mu\text{g}/\text{m}^3$) | Val $R^2$ | Train Time (s) | Validation Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | **0.03** | **5** | **0.90** | **41.59** | **57.16** | **0.1392** | 8.29 | **1 (Best $R^2$ / RMSE)** |
| 1 | 0.03 | 5 | 0.75 | 41.59 | 57.27 | 0.1359 | 8.56 | 2 |
| 10 | 0.06 | 7 | 0.90 | 41.68 | 57.29 | 0.1354 | 5.97 | 3 |
| 8 | 0.06 | 5 | 0.90 | 41.60 | 57.34 | 0.1337 | 5.25 | 4 |
| 7 | 0.06 | 5 | 0.75 | 41.77 | 57.38 | 0.1324 | 4.68 | 5 |
| 14 | 0.10 | 5 | 0.90 | 41.47 | 57.43 | 0.1311 | 5.07 | 6 (Lowest MAE) |
| 13 | 0.10 | 5 | 0.75 | 41.70 | 57.50 | 0.1288 | 3.95 | 7 |
| 4 | 0.03 | 7 | 0.90 | 42.02 | 57.51 | 0.1286 | 8.95 | 8 |
| 3 | 0.03 | 7 | 0.75 | 41.85 | 57.54 | 0.1277 | 10.07 | 9 |
| 9 | 0.06 | 7 | 0.75 | 41.75 | 57.57 | 0.1269 | 6.04 | 10 |
| 16 | 0.10 | 7 | 0.90 | 41.82 | 57.59 | 0.1262 | 4.41 | 11 |
| 15 | 0.10 | 7 | 0.75 | 41.74 | 57.62 | 0.1251 | 5.10 | 12 |
| 12 | 0.06 | 9 | 0.90 | 42.46 | 57.72 | 0.1223 | 7.14 | 13 |
| 8 | 0.03 | 9 | 0.90 | 42.27 | 57.79 | 0.1200 | 11.20 | 14 |
| 18 | 0.10 | 9 | 0.90 | 42.37 | 57.79 | 0.1200 | 5.04 | 15 |
| 11 | 0.06 | 9 | 0.75 | 42.08 | 57.90 | 0.1169 | 8.28 | 16 |
| 5 | 0.03 | 9 | 0.75 | 42.24 | 57.97 | 0.1145 | 12.03 | 17 |
| 17 | 0.10 | 9 | 0.75 | 43.03 | 58.06 | 0.1120 | 4.54 | 18 (Worst Overfit) |

---

## 3. Mathematical Analysis & Response Surface Dynamics

```
                     MAX DEPTH VS VALIDATION R^2
       Val R^2
       0.140 +    * (d=5, lr=0.03, s=0.90) -> Peak Generalisation (R^2 = 0.1392)
       0.135 +   *** (d=5 cluster)
       0.130 +       *** (d=7 cluster)
       0.125 +
       0.120 +           *** (d=9 cluster)
       0.115 +              * (d=9, lr=0.10, s=0.75) -> Overfitting Floor (R^2 = 0.1120)
       0.110 +---------------------------------------------------
                  d=5             d=7             d=9 (Max Tree Depth)
```

### 3.1 Depth-Variance Trade-Off and the Regularization of Shallow Trees
A decisive finding is the **monotonic inverse relationship between tree depth and validation generalization**:
- **Shallow Trees ($d=5$):** Dominate the top tier of validation performance ($R^2 \in [0.1288, 0.1392]$). By restricting tree depth to 5 (at most 32 terminal leaves per tree), the learner is constrained to low-order feature interactions (e.g., wind speed interacted with dew point depression, or temperature interacted with boundary layer pressure).
- **Deep Trees ($d=9$):** Exhibit severe validation degradation ($R^2 \in [0.1120, 0.1223]$). At depth 9 (up to 512 leaves per tree), the model fragments the sample space into hyper-localized micro-regimes, fitting spurious high-frequency turbulence present in the 2013–2015 training set that fails to generalize across 2016 validation seasons.

### 3.2 Learning Rate and Gradient Step Dynamics
- A conservative shrinkage factor ($\eta = 0.03$) with 200 early-stopped trees delivers the highest statistical stability ($R^2 = 0.1392$), allowing smooth descent on the non-convex loss surface.
- A higher learning rate ($\eta = 0.10$) trains in under $5\text{s}$ and achieves slightly lower absolute MAE ($41.47\ \mu\text{g}/\text{m}^3$), but exhibits larger squared error tails (RMSE $57.43\ \mu\text{g}/\text{m}^3$), indicating occasional sensitivity to extreme outliers.

### 3.3 Subsampling Regularization
- Subsample ratio $s = 0.90$ consistently outperformed $s = 0.75$ across nearly all depth and learning rate pairs. Given the large training sample volume ($N = 291,574$), retaining $90\%$ of observations provides low-variance gradient estimates while preserving sufficient stochasticity to de-correlate consecutive boosting rounds.

---

## 4. Final Unbiased Test Split Evaluation

The optimal configuration selected on the validation partition:
$$\boldsymbol{\theta}^* = \{\eta = 0.03, d = 5, s = 0.90\}$$
was evaluated on the strictly held-out test split (October 2, 2016 – February 28, 2017; 42,278 hourly observations).

### Quantitative Comparison: Default vs. Tuned Forecaster ($h=24\text{h}$)

| Metric | Default XGBoost (Exp 1) | Tuned XGBoost (Exp 4) | Relative Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Test MAE ($\mu\text{g}/\text{m}^3$)** | 67.26 | **66.29** | **$-0.97\ \mu\text{g}/\text{m}^3$ ($-1.4\%$)** |
| **Test RMSE ($\mu\text{g}/\text{m}^3$)** | 93.41 | **92.15** | **$-1.26\ \mu\text{g}/\text{m}^3$ ($-1.3\%$)** |
| **Test $R^2$** | 0.1978 | **0.2193** | **$+0.0215$ ($+10.9\%$ rel.)** |
| **Normal MAE ($y < 150$)** | 50.61 | **49.43** | **$-1.18\ \mu\text{g}/\text{m}^3$ ($-2.3\%$)** |
| **Extreme MAE ($y \ge 150$)** | 119.40 | **119.09** | **$-0.31\ \mu\text{g}/\text{m}^3$** |
| **Pearson Correlation ($r$)** | 0.4451 | **0.4722** | **$+0.0271$ ($+6.1\%$ rel.)** |
| **sMAPE (%)** | 78.11% | **77.52%** | **$-0.59\%$** |
| **MAPE (%)** | 330.27% | **323.74%** | **$-6.53\%$** |

### Key Takeaways from Test Generalization
1. **Unbiased Skill Transfer:** The hyperparameter improvements observed on the 2016 validation partition transferred seamlessly to the unseen 2016–2017 winter test set, boosting test $R^2$ from $0.1978$ to $0.2193$.
2. **Sub-Severe Baseline Improvement:** The primary driver of test MAE reduction was in the typical/moderate regime ($y < 150\ \mu\text{g}/\text{m}^3$), where MAE improved from $50.61$ to $49.43\ \mu\text{g}/\text{m}^3$. Shallow depth prevented over-reacting to mild meteorological fluctuations during non-stagnant periods.
3. **Correlation Gain:** Pearson correlation surged from $0.4451$ to $0.4722$, proving that the regularized model more accurately tracks phase shifts and temporal transitions in 24-hour pollutant cycles.
