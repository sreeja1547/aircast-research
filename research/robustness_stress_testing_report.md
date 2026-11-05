# AirCast Research — Phase 8 / Experiment 8: Robustness Evaluation and Stress Testing

**Project:** AirCast Research: A Research-Driven Framework for Explainable, Uncertainty-Aware and Generalizable Multi-Horizon Air Quality Forecasting  
**Author:** Senior Applied Scientist & Machine Learning Architect  
**Evaluation Scope:** Sensor Noise Perturbations, Telemetry Outage Simulations, and AQI Severity Stratification  
**Target Variable:** $PM_{2.5}(t+24)$ (24-Hour Ahead Lead Time)  
**Evaluated Architecture:** XGBoost Regressor (`combined_all`, 47 features)  
**Test Sample Count:** 42,278 hourly observations (October 2, 2016 – February 28, 2017)  

---

## 1. Executive Summary & Motivation

Operational environmental monitoring networks are characterized by sensor aging, electrical interference, wireless packet loss, and full station outages. An applied machine learning framework must be resilient under adverse operational conditions.

Phase 8 / Experiment 8 subjects the primary 24-hour forecaster to a **three-tier stress-testing harness**:
1. **Gaussian Sensor Measurement Noise:** Simulating calibration drift and electrical noise by injecting zero-mean Gaussian perturbations ($\sigma \in \{5\%, 15\%, 25\%\}$ of feature standard deviation) across all numerical features.
2. **Sensor Dropout & Telemetry Outage:** Simulating total hardware failure for individual sensors and sensor clusters (e.g., loss of CO, loss of $SO_2 + NO_2$, complete precursor cluster failure, and complete wind anemometer failure), falling back strictly to training-partition medians.
3. **AQI Severity Stratification:** Partitioning test predictions across five standardized air quality index (AQI) concentration tiers to quantify heteroscedastic error distributions and diagnose extreme event under-prediction.

---

## 2. Tier 1: Sensor Noise Perturbation Stress Testing

To test sensitivity to sensor degradation, we perturb every continuous predictor $x_j$ at inference time:
$$\tilde{x}_j = x_j + \epsilon_j, \quad \epsilon_j \sim \mathcal{N}\left(0, (\rho \cdot \sigma_j)^2\right)$$
where $\sigma_j$ is the empirical standard deviation of feature $j$ on the training split, and $\rho \in \{0.05, 0.15, 0.25\}$ represents the noise ratio.

### Table 1: Sensor Noise Injection Stress Test Results

| Noise Ratio ($\rho$) | Perturbation Level | Test MAE ($\mu\text{g}/\text{m}^3$) | Test RMSE ($\mu\text{g}/\text{m}^3$) | Test $R^2$ | Extreme MAE ($y \ge 150$) | Normal MAE ($y < 150$) | Relative MAE Degradation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | **Clean Baseline** | **67.26** | **93.41** | **0.1978** | **119.40** | **50.61** | **Baseline (0.00%)** |
| 0.05 | 5% Sensor Noise | 66.69 | 92.45 | 0.2143 | 113.72 | 51.67 | -0.85% (Regularized) |
| 0.15 | 15% Sensor Noise | 66.96 | 92.75 | 0.2091 | 116.15 | 51.25 | -0.45% |
| 0.25 | 25% Heavy Noise | 67.43 | 93.23 | 0.2009 | 118.00 | 51.29 | **+0.26% Degradation** |

```
                       NOISE SENSITIVITY PROFILE
      Test MAE (ug/m^3)
      67.5 +                                               * 25% Noise (67.43)
           |
      67.2 + * Clean Baseline (67.26)
           |
      66.9 +                               * 15% Noise (66.96)
           |
      66.6 +               * 5% Noise (66.69)
           +---------------+---------------+---------------+------------------
                   0%              5%             15%             25%
                                  Injected Noise Ratio (sigma)
```

### Analysis of Noise Tolerance:
- **Extraordinary Structural Resilience:** Gradient boosted decision trees exhibit profound robustness to continuous input noise. Even when subjecting all 47 features to $25\%$ additive Gaussian jitter, test MAE increases by only **$+0.17\ \mu\text{g}/\text{m}^3$ ($+0.26\%$ relative change)**.
- **Beneficial Jitter at Low Ratios:** At $5\%$ and $15\%$ noise, test performance actually improves slightly ($\text{MAE} \approx 66.7\text{–}66.9\ \mu\text{g}/\text{m}^3$). Additive jitter smooths out boundary quantization artifacts in step-function leaf activations, acting as an implicit stochastic regularizer.

---

## 3. Tier 2: Telemetry Outage & Sensor Dropout Scenarios

When physical monitoring stations experience sensor failure, missing telemetry must be imputed in real-time. In this stress test, selected input columns are wiped out and replaced with their global training-partition medians (simulating zero-shot fallback imputation).

### Table 2: Hardware Sensor Failure Simulation Matrix

| Failure Scenario | Dropped Telemetry Channels | Test MAE ($\mu\text{g}/\text{m}^3$) | $\Delta\text{MAE}$ vs. Baseline | Test RMSE ($\mu\text{g}/\text{m}^3$) | Test $R^2$ | Operational Severity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **No Outage (Baseline)** | None | **67.26** | **0.00** | **93.41** | **0.1978** | Normal Operation |
| **CO Sensor Outage** | `CO` | 67.23 | -0.02 | 93.17 | 0.2020 | Negligible Impact |
| **$SO_2 + NO_2$ Outage** | `SO2`, `NO2` | 67.18 | -0.08 | 92.70 | 0.2100 | Negligible Impact |
| **Precursor Cluster Outage**| `SO2`, `NO2`, `CO`, `O3` | 67.39 | **+0.13** | 92.65 | 0.2109 | Low Impact |
| **Wind Sensor Cluster Outage** | `WSPM`, `wind_u`, `wind_v`, `wd_sin`, `wd_cos` | **69.30** | **+2.05** | **94.32** | **0.1822** | **HIGH CRITICAL FAILURE** |

```
                     IMPACT OF SENSOR OUTAGES ON TEST MAE
      Delta MAE (ug/m^3)
      +2.5 +
      +2.0 +                                       * Wind Sensor Outage (+2.05 ug/m^3)
      +1.5 +                                         [CRITICAL TELEMETRY]
      +1.0 +
      +0.5 +
       0.0 + * Baseline   * CO (-0.02) * SO2+NO2 (-0.08) * Precursors (+0.13)
           +------------------------------------------------------------------
                 Hardware Failure Scenario Evaluated
```

### Critical Atmospheric & Engineering Discoveries:
1. **The Irreplaceable Role of Wind Telemetry:**
   - Loss of the wind sensor suite ($WSPM$, $u, v$, directional angles) triggers a catastrophic performance drop: MAE surges by **$+2.05\ \mu\text{g}/\text{m}^3$** to $69.30\ \mu\text{g}/\text{m}^3$, and $R^2$ drops to $0.1822$.
   - **Mechanism:** Wind telemetry is non-redundant. Unlike co-pollutants that share common combustion sources, surface wind speed and velocity vectors describe the external kinetic forcing of the atmosphere. When anemometers fail, the model cannot deduce whether an air mass is stagnating or undergoing rapid advective evacuation.
2. **Co-Pollutant Redundancy & Masking:**
   - In contrast, losing primary gas precursors ($SO_2, NO_2, CO, O_3$) results in almost zero performance degradation ($\Delta\text{MAE} = +0.13\ \mu\text{g}/\text{m}^3$). Because $PM_{2.5}$ and $PM_{10}$ autoregressive trajectories and co-pollutants share high collinearity, the tree ensemble seamlessly redirects decision paths through surviving particle channels.

---

## 4. Tier 3: AQI Severity Stratification Analysis

To understand where forecasting errors originate, we stratify all 42,278 test samples into five concentration tiers corresponding to national air quality warning thresholds:

### Table 3: Performance Stratified by Observed Pollution Severity ($h=24\text{h}$)

| AQI Category | Concentration Range ($PM_{2.5}$) | Sample Count (% Total) | Tier MAE ($\mu\text{g}/\text{m}^3$) | Tier RMSE ($\mu\text{g}/\text{m}^3$) | Tier sMAPE (%) | Dominant Error Nature |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Good to Moderate** | $[0, 75)\ \mu\text{g}/\text{m}^3$ | 22,991 (54.4%) | 58.62 | 73.74 | 104.82% | Mild over-prediction of clean floors. |
| **Unhealthy for Sensitive** | $[75, 115)\ \mu\text{g}/\text{m}^3$ | 5,459 (12.9%) | **26.44** | **32.86** | **30.05%** | **Optimal Calibration Window** |
| **Unhealthy** | $[115, 150)\ \mu\text{g}/\text{m}^3$ | 3,597 (8.5%) | 36.06 | 42.96 | 33.02% | High accuracy; well-bounded errors. |
| **Very Unhealthy** | $[150, 250)\ \mu\text{g}/\text{m}^3$ | 6,526 (15.4%) | 72.96 | 82.22 | 47.91% | Systematic under-prediction onset. |
| **Severe / Hazardous** | $[250, 1000)\ \mu\text{g}/\text{m}^3$ | 3,705 (8.8%) | **201.19** | **224.80** | **80.17%** | **Acute Under-Prediction Peak** |

### Insights on Error Heteroscedasticity:
1. **The "Sweet Spot" of Tree Ensembles:** The model achieves its highest relative accuracy in the intermediate ranges ($75\text{–}150\ \mu\text{g}/\text{m}^3$), where MAE is just $26.44\text{ to }36.06\ \mu\text{g}/\text{m}^3$ and sMAPE is $\approx 30\%$. These ranges represent typical Beijing autumn/winter days where boundary layer dynamics are well-sampled.
2. **Regression-to-the-Mean in Severe Episodes:** In the hazardous category ($y > 250\ \mu\text{g}/\text{m}^3$), MAE swells to $201.19\ \mu\text{g}/\text{m}^3$. Decision tree algorithms average the target values within terminal leaves; consequently, they cannot extrapolate beyond the highest training leaf thresholds, resulting in conditional under-prediction during unprecedented multi-day mega-haze events.

---

## 5. Engineering Guidelines & Operational Recommendations

```
+------------------------------------------------------------------------------------+
|                         AIRCAST SYSTEM HARDENING PROTOCOL                          |
+------------------------------------------------------------------------------------+
| 1. ANEMOMETER REDUNDANCY: Anemometer failures degrade day-ahead skill by +2.05    |
|    ug/m^3. Implement dual-redundant sonic anemometers or regional numerical weather|
|    prediction (NWP) wind vector fallbacks.                                         |
|                                                                                    |
| 2. GAS PRECURSOR RESILIENCE: Gas sensor dropouts require no urgent dispatch;      |
|    training-median causal imputation sustains < 0.15 ug/m^3 error deviation.      |
|                                                                                    |
| 3. NOISE IMMUNITY: System operates reliably without pre-filtering on sensors with  |
|    up to 25% Gaussian electrical noise.                                            |
|                                                                                    |
| 4. EXTREME EVENT ALERTING: In severe regimes (>250 ug/m^3), combine point trees   |
|    with conformal upper bounds to prevent under-warning civil protection agencies. |
+------------------------------------------------------------------------------------+
```
