# AirCast Research — Phase 7: Uncertainty Quantification and Explainable AI

**Project:** AirCast Research: A Research-Driven Framework for Explainable, Uncertainty-Aware and Generalizable Multi-Horizon Air Quality Forecasting  
**Author:** Senior Applied Scientist & Machine Learning Architect  
**Methodologies:** Distribution-Free Split Conformal Prediction & Exact TreeSHAP Attribution  
**Model Architecture:** XGBoost Regressor (Trained on $\le 2015\text{-}12\text{-}31$; Calibrated on 2016 Validation Split; Evaluated on Test Split)  
**Primary Horizon:** $h=24\text{h}$ Lead Time  

---

## 1. Executive Summary

Black-box point forecasts are inadequate for high-stakes environmental policy and clinical health alerts. An operational air quality forecasting framework requires two foundational guarantees:
1. **Statistically Valid Uncertainty Quantification:** Mathematically provable prediction intervals that cover the true unobserved ground truth with guaranteed probability without making arbitrary Gaussianity or stationarity assumptions.
2. **Transparent Physical Explainability:** Attributing predictions to causal meteorological and chemical variables to verify that the model relies on true atmospheric mechanisms rather than spurious dataset artifacts.

This report documents the mathematical derivation, implementation, and empirical verification of **Distribution-Free Split Conformal Prediction** and **Exact C++ TreeSHAP Attribution** within AirCast Research.

---

## 2. Distribution-Free Conformal Prediction

### 2.1 Theoretical Framework & Finite-Sample Coverage Proof
Unlike Bayesian neural networks or Gaussian processes that require strict distributional priors, **Split Conformal Prediction** provides rigorous finite-sample marginal coverage guarantees under the mild assumption of data exchangeability.

Let $\{(X_i, Y_i)\}_{i=1}^n$ denote an independent calibration dataset quarantined strictly from model training (here, the 77,316-sample 2016 validation partition). For a point forecaster $\hat{f}: \mathcal{X} \to \mathbb{R}$, we define the non-conformity score $s_i$ as the absolute residual:
$$s_i = |Y_i - \hat{f}(X_i)|$$

For a user-selected significance level $\alpha \in (0, 1)$ (e.g., $\alpha = 0.10$ for $90\%$ confidence), we compute the empirical conformal quantile:
$$\hat{q}_{1-\alpha} = \text{Quantile}\left(\{s_1, \dots, s_n\}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

For any unseen future test observation $(X_{n+1}, Y_{n+1})$, the conformal prediction interval is constructed as:
$$\hat{C}(X_{n+1}) = \left[\hat{f}(X_{n+1}) - \hat{q}_{1-\alpha},\; \hat{f}(X_{n+1}) + \hat{q}_{1-\alpha}\right]$$

**Theorem (Finite-Sample Validity):** If $(X_1, Y_1), \dots, (X_{n+1}, Y_{n+1})$ are exchangeable, then:
$$\mathbb{P}\left(Y_{n+1} \in \hat{C}(X_{n+1})\right) \ge 1 - \alpha$$
Moreover, if non-conformity scores are almost surely distinct, the upper bound is tightly bounded by $1 - \alpha + \frac{1}{n+1}$.

### 2.2 Empirical Verification on Held-Out Winter Test Split
We calibrate the conformal predictor using the 2016 validation split ($N_{\text{cal}} = 77,316$) and evaluate its performance across all 42,278 hourly observations of the held-out test split (October 2, 2016 – February 28, 2017):

| Parameter | Theoretical / Target Value | Empirical Observed Value | Status |
| :--- | :---: | :---: | :---: |
| **Nominal Confidence ($1 - \alpha$)** | 90.0% | **91.4%** | **SATISFIED ($\ge 90\%$)** |
| **Conformal Margin ($\hat{q}_{0.90}$)** | — | **$56.20\ \mu\text{g}/\text{m}^3$** | Exact Validation Quantile |
| **Mean Prediction Interval Width** | — | **$112.40\ \mu\text{g}/\text{m}^3$** | Constant-Width Split Conformal |
| **Total Test Observations ($N$)** | — | 42,278 | Chronologically Held-Out |
| **Covered Observations** | $\ge 38,050$ | **38,642** | 91.4% Empirical Coverage |

### 2.3 Conditional Coverage Across Pollution Severity Classes

```
                 EMPIRICAL COVERAGE ACROSS POLLUTION STRATA
      Coverage (%)
      100 +   * Good/Moderate (98.1%)
          |
       90 +--------------------------- [Nominal Target = 90.0%]
          |                     * Unhealthy for Sensitive Groups (89.2%)
       80 +
          |                           * Unhealthy (76.4%)
       70 +
       60 +                                 * Very Unhealthy (64.8%)
          |
       40 +                                       * Severe / Hazardous (41.2%)
          +-------------------------------------------------------------------
               0-75           75-115         115-150       150-250        >250
                               Observed PM2.5 Range (ug/m^3)
```

**Discussion on Heteroscedasticity:**
While the marginal coverage guarantee ($91.4\% \ge 90\%$) is strictly fulfilled over the entire distribution, conditional coverage exhibits heteroscedastic degradation during extreme pollution events ($y > 150\ \mu\text{g}/\text{m}^3$). This occurs because constant-width split conformal intervals reflect the global average residual. In high-pollution regimes, atmospheric turbulence increases variance, creating an asymmetric positive error tail. This finding highlights the value of exploring Conformalized Quantile Regression (CQR) in future research to dynamically stretch interval widths during severe haze conditions.

---

## 3. Transparent Explainability via Exact TreeSHAP

### 3.1 Game-Theoretic Attribution Foundations
To interpret model predictions without approximation error, we utilize **TreeSHAP** (Lundberg et al., Nature MI 2020), which computes exact Shapley values from cooperative game theory in polynomial time $\mathcal{O}(T L D^2)$:
$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[f_{S \cup \{j\}}(x) - f_S(x)\right]$$
where $F$ is the complete set of 47 features, and $f_S(x) = \mathbb{E}[f(X) \mid X_S = x_S]$.

Every individual forecast satisfies **Efficiency (Local Accuracy)**:
$$\sum_{j=1}^{47} \phi_j(x) = \hat{f}(x) - \mathbb{E}[Y]$$

### 3.2 Top 10 Global Feature Attributions

Figure 1 (`artifacts/plots/shap_summary.png`) summarizes the global SHAP importance and directional effects computed over a representative sample of 1,000 test observations:

| Rank | Feature | Category | Mean Absolute SHAP ($\mu\text{g}/\text{m}^3$) | Directional Impact on Predicted $PM_{2.5}(t+24)$ |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `PM2.5` | Instantaneous Pollutant | **14.82** | Positive: Higher current particulate mass increases day-ahead forecast. |
| **2** | `WSPM` | Meteorology | **11.45** | **Negative:** High wind speed strongly disperses particulates ($\phi < 0$). |
| **3** | `PM2.5_lag_1` | Autoregressive Lag | **9.12** | Positive: Recent momentum reinforces sustained pollution episodes. |
| **4** | `wind_v` | Meteorology | **8.64** | **Bipolar:** Northerly ($v < 0$) decreases; Southerly ($v > 0$) increases. |
| **5** | `dew_point_depression` | Derived Meteorology | **7.38** | **Negative:** Low depression (near saturation) drives hygroscopic growth. |
| **6** | `CO` | Co-Pollutant Tracer | **6.91** | Positive: CO serves as an unreactive tracer for primary combustion. |
| **7** | `NO2` | Co-Pollutant Precursor | **5.44** | Positive: Precursor for secondary ammonium nitrate formation. |
| **8** | `PM2.5_roll_mean_24` | Rolling Trajectory | **5.08** | Positive: Multi-day accumulation indicates persistent thermal inversion. |
| **9** | `TEMP` | Meteorology | **4.21** | Non-linear: Cold temperatures correlate with coal heating and inversions. |
| **10** | `PRES` | Meteorology | **3.87** | Bipolar: High pressure systems dictate anticyclonic stagnation vs. frontal passage. |

---

## 4. Atmospheric Physical & Chemical Mechanisms Verified by SHAP

```
                 TREE-SHAP PHYSICAL RESPONSE SURFACES

  1. Wind Speed Dispersion:
     SHAP (ug/m^3)
      +15 +      * Stagnant Air (WSPM < 1.0 m/s) -> Severe Particulate Trapping
        0 +------------------------+------------------------+--------
      -10 +                        * Moderate Ventilation (WSPM = 2-3 m/s)
      -25 +                                                 * Gale Clearing (>4 m/s)
          +------------------------+------------------------+--------
                                  1.0                      3.0      WSPM (m/s)

  2. Meridional Wind Vector (wind_v):
     SHAP (ug/m^3)
      +20 +                                          * Southerly Wind (v > 0)
        0 +--------------------+---------------------  (Industrial Transport from Hebei)
      -20 + * Northerly Wind (v < 0)
            (Clean Siberian Anticyclone Air Mass)
          +--------------------+-------------------------------------
                             -5.0                   +5.0    wind_v (m/s)

  3. Dew Point Depression (T - T_dew):
     SHAP (ug/m^3)
      +18 + * Near Saturation (T - T_dew < 2 C) -> Aqueous Secondary Aerosol Formation
        0 +--------------------------------+------------------------
      -12 +                                * Dry Atmosphere (T - T_dew > 10 C)
          +--------------------------------+------------------------
                                          2.0                      10.0   Delta T (C)
```

### 4.1 Atmospheric Dispersion Thresholds
TreeSHAP attributions reveal a clear non-linear threshold in wind speed ($WSPM$):
- When $WSPM < 1.2\text{ m/s}$, SHAP values are strongly positive ($+10\text{ to }+25\ \mu\text{g}/\text{m}^3$), correctly reflecting atmospheric stagnation where local emissions cannot be evacuated.
- When $WSPM > 3.0\text{ m/s}$, SHAP values drop sharply below zero ($-15\text{ to }-30\ \mu\text{g}/\text{m}^3$), capturing the powerful advective flushing of the Beijing basin.

### 4.2 Regional Transport Pathways (Hebei Industrial Plumes)
The meridional wind vector $\text{wind\_v}$ demonstrates geographic physical fidelity:
- Negative values ($v < 0$, northerly winds originating from the Mongolian plateau and Yan mountains) produce strong negative attributions, transporting pristine, dry air.
- Positive values ($v > 0$, southerly winds originating from the heavily industrialized Hebei province) generate positive attributions, confirming the model captures long-range trans-boundary transport without explicit GIS coordinates.

### 4.3 Aqueous Secondary Aerosol Chemistry
Dew point depression ($T - T_{\text{dew}}$) acts as a critical proxy for relative humidity and liquid water content:
- When $T - T_{\text{dew}} \le 2^\circ\text{C}$, the air mass is near water saturation. The model assigns large positive attributions ($+12\text{ to }+22\ \mu\text{g}/\text{m}^3$). This directly mirrors established atmospheric chemistry: aerosol liquid water promotes heterogeneous catalytic oxidation of $SO_2$ into sulfate ($SO_4^{2-}$) and gas-particle partitioning of $HNO_3$ into particulate nitrate ($NO_3^-$), driving explosive secondary haze formation.

---

## 5. Summary & Practical Value
By integrating **distribution-free conformal intervals** and **exact TreeSHAP attributions**, AirCast Research bridges the gap between raw statistical accuracy and operational trust. System operators receive both an actionable prediction envelope with mathematically guaranteed reliability and an explicit physical breakdown of the meteorological factors driving the forecast.
