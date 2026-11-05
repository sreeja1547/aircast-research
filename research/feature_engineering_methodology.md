# AirCast Research: Advanced Feature Engineering & Multi-Horizon Pipeline

**Document Version:** 1.0.0  
**Module Reference:** `src/features/` (`builder.py`, `registry.py`) and `src/data/` (`forecasting_dataset.py`)  
**Test Suite:** `tests/test_feature_engineering.py` (14 total tests passed across test suites)  

---

## 1. Problem Setting & Multi-Horizon Formulation

Air quality forecasting is framed as a supervised multi-horizon direct regression problem. For each station $s \in \{1, \dots, 12\}$ and discrete time index $t$, the system predicts future $PM_{2.5}$ concentrations at three distinct lead times:

$$h \in \{1, 6, 24\}\ \text{hours}$$

$$\hat{y}_s^{(h)}(t) = f_h\left( \mathbf{X}_s(t) \right)$$

where $\mathbf{X}_s(t)$ represents the feature representation constructed **strictly from historical observations at or prior to time $t$**.

### Target Label Alignment & Absolute Target Integrity
Target labels are defined as:
$$y_s^{(1)}(t) = y_s^{\text{raw}}(t + 1)$$
$$y_s^{(6)}(t) = y_s^{\text{raw}}(t + 6)$$
$$y_s^{(24)}(t) = y_s^{\text{raw}}(t + 24)$$

**Guarantees:**
1. Ground truth target labels $y_s^{(h)}(t)$ are derived **strictly from the raw, un-imputed measurements**.
2. If $y_s^{\text{raw}}(t + h)$ is unobserved (`NaN`), the instance $(\mathbf{X}_s(t), y_s^{(h)}(t))$ is excluded from training and evaluation.
3. No synthetic or forward-filled values are ever substituted for target labels.

---

## 2. Mathematical Definition of Feature Groups

The feature space $\mathbf{X}_s(t)$ is partitioned into modular, hypothesis-driven feature groups:

```
                          [ Raw Historical Observations at time <= t ]
                                               │
     ┌──────────────────┬──────────────────────┼───────────────────────┬──────────────────┐
     ▼                  ▼                      ▼                       ▼                  ▼
[ Temporal ]     [ Instantaneous ]      [ Autoregressive ]       [ Causal Rolling ] [ Meteorological ]
- hour_sin/cos   - PM2.5(t)             - PM2.5(t-1)             - mean_[3,6,12,24] - TEMP, PRES, DEWP
- month_sin/cos  - PM10(t), SO2(t)      - PM2.5(t-3), PM2.5(t-6) - std_[3,6,12,24]  - RAIN, has_rain
- day_of_week    - NO2(t), CO(t), O3(t) - PM2.5(t-12)            - min_[3,6,12,24]  - WSPM, u, v
- is_weekend                            - PM2.5(t-24), (t-48)    - max_[3,6,12,24]  - dew_depression
- is_heating                                                                        - wd_sin, wd_cos
```

### Group A: Autoregressive Lag Features
Autoregressive lags capture historical temporal persistence and inertia in atmospheric aerosol mass:

$$L_k(t) = y_s(t - k), \quad k \in \{1, 3, 6, 12, 24, 48\}\ \text{hours}$$

- **1-Hour Lag ($k=1$):** Captures high-frequency autoregressive inertia.
- **3-Hour and 6-Hour Lags ($k=3, 6$):** Captures synoptic accumulation/dispersion onset trends.
- **12-Hour Lag ($k=12$):** Captures diurnal inversion transitions (e.g., matching midnight accumulation against midday dilution).
- **24-Hour Lag ($k=24$):** Captures diurnal periodicity and seasonal persistence (same hour yesterday).
- **48-Hour Lag ($k=48$):** Captures multi-day stagnation cycles.

### Group B: Causal Multi-Window Rolling Statistics
Rolling statistics capture the local atmospheric trajectory, accumulation velocity, and dispersion volatility:

For window size $W \in \{3, 6, 12, 24\}$ hours, computed causally over $[t - W + 1, t]$:

1. **Rolling Mean:**
   $$\mu_{W}(t) = \frac{1}{W} \sum_{i=0}^{W-1} y_s(t - i)$$
   *Physical Meaning:* Smoothed particulate background level, filtering out high-frequency instrument noise.

2. **Rolling Standard Deviation:**
   $$\sigma_{W}(t) = \sqrt{\frac{1}{W - 1} \sum_{i=0}^{W-1} \left( y_s(t - i) - \mu_{W}(t) \right)^2}$$
   *Physical Meaning:* Atmospheric volatility; spikes indicate rapid transitions (e.g., arrival of a clean front or rapid onset of inversion).

3. **Rolling Minimum:**
   $$\min_{W}(t) = \min \{ y_s(t - i) : 0 \le i < W \}$$
   *Physical Meaning:* Baseline regional background level beneath localized emission spikes.

4. **Rolling Maximum:**
   $$\max_{W}(t) = \max \{ y_s(t - i) : 0 \le i < W \}$$
   *Physical Meaning:* Peak pollution severity experienced during the accumulation window.

### Group C: Periodic Cyclical Encodings
Because hour of day and month of year are circular (hour 23 is 1 hour away from hour 0; December is 1 month away from January), linear scalar encoding introduces false discontinuities. They are mapped to the 2D Cartesian unit circle:

$$\text{hour}_{\sin}(t) = \sin\left(\frac{2\pi \cdot \text{hour}(t)}{24}\right), \quad \text{hour}_{\cos}(t) = \cos\left(\frac{2\pi \cdot \text{hour}(t)}{24}\right)$$

$$\text{month}_{\sin}(t) = \sin\left(\frac{2\pi \cdot (\text{month}(t) - 1)}{12}\right), \quad \text{month}_{\cos}(t) = \cos\left(\frac{2\pi \cdot (\text{month}(t) - 1)}{12}\right)$$

### Group D: Surface Meteorology & Physical Vector Decompositions
1. **Cartesian Wind Vectors:** Wind direction is measured in degrees $\theta \in [0^\circ, 360^\circ)$. It is decomposed into orthogonal physical components:
   $$u(t) = -WSPM(t) \cdot \sin\left(\frac{2\pi \cdot \theta(t)}{360}\right) \quad (\text{Zonal wind: positive eastward})$$
   $$v(t) = -WSPM(t) \cdot \cos\left(\frac{2\pi \cdot \theta(t)}{360}\right) \quad (\text{Meridional wind: positive northward})$$
   $$\text{wd}_{\sin}(t) = \sin\left(\frac{2\pi \cdot \theta(t)}{360}\right), \quad \text{wd}_{\cos}(t) = \cos\left(\frac{2\pi \cdot \theta(t)}{360}\right)$$

2. **Dew Point Depression ($\Delta_{\text{dew}}$):**
   $$\Delta_{\text{dew}}(t) = TEMP(t) - DEWP(t) \quad (^\circ\text{C})$$
   *Physical Meaning:* Proxy for relative humidity ($RH \approx 100 - 5 \Delta_{\text{dew}}$). Low depression signals near-saturation, driving secondary aqueous inorganic aerosol synthesis.

3. **Precipitation Washout Indicator:**
   $$\text{has\_rain}(t) = \mathbb{I}(RAIN(t) > 0.0)$$
   *Physical Meaning:* Binary indicator for wet atmospheric scavenging, accounting for the 95.98% dry sparsity.

---

## 3. Controlled Experimental Feature Registries

To support ablation studies (Experiment 2 and Experiment 3 in Phase 5), features are registered in `src/features/registry.py`:

| Feature Group Identifier | Number of Features | Included Feature Signals | Research Question Addressed |
| :--- | :-: | :--- | :--- |
| `temporal_baseline` | 8 | $\text{hour}_{\sin/\cos}, \text{month}_{\sin/\cos}, \text{day\_of\_week}, \text{is\_weekend}, \text{is\_heating}, \text{station}$ | Establishes lower bound performance from calendar seasonality alone. |
| `historical_pollutants`| 14 | Temporal baseline + instantaneous $PM_{2.5}, PM_{10}, SO_2, NO_2, CO, O_3$ at time $t$ | Evaluates baseline multivariate co-pollutant information without memory. |
| `lag_features` | 20 | Historical pollutants + $PM_{2.5}$ lags at $1, 3, 6, 12, 24, 48$ hours | **RQ2:** Quantifies the marginal predictive gain of autoregressive memory. |
| `rolling_features` | 36 | Lag features + rolling means, std, min, max ($3, 6, 12, 24$ hours) | **RQ2:** Quantifies the benefit of short/medium-term trajectory statistics. |
| `meteorological_features`| 25 | Temporal + pollutants + $TEMP, PRES, DEWP, RAIN, \text{has\_rain}, WSPM, \Delta_{\text{dew}}, u, v, \text{wd}_{\sin/\cos}$ | **RQ3:** Isolates the contribution of synoptic weather and wind transport. |
| `combined_all` | 47 | Full multimodal integration across all above feature sets | Primary input representation for multi-horizon model benchmarking. |

---

## 4. Potential Data Leakage Vectors and Verified Defenses

| Potential Leakage Vector | Mechanism of Failure | AirCast Engineered Defense | Verification Test |
| :--- | :--- | :--- | :--- |
| **Cross-Station Lag Bleed** | Lag feature at index 0 of Station B accessing the last row of Station A | Features computed strictly within `groupby('station')` groups | `test_station_boundary_isolation` |
| **Rolling Window Peeking** | Centered rolling windows including $t+1, \dots, t+k$ | Strictly causal forward windows over $[t-W+1, t]$; zero lookahead | `test_rolling_window_no_lookahead` |
| **Target Imputation Leakage** | Target label at $t+h$ being filled by forward-fill imputer | Targets derived from raw un-imputed series prior to feature imputation | `test_unimputed_target_integrity` |
| **Future Weather Lookahead** | Using actual observed weather at $t+h$ as a feature | Only weather observed $\le t$ enters feature vector $\mathbf{X}_s(t)$ | Verified by construction |
| **LOSO Spatial Contamination** | Using held-out station data during imputer fitting | Unseen stations fall back strictly to training set global statistics | `test_loso_imputer_isolation` |
| **Purged Split Overlap** | Target $y(t+24)$ spilling across train-val split boundary | 24-hour embargo buffer between splits | `test_temporal_purged_splitter` |

---

## 5. Automated Test Suite Results

The combined test suite encompassing preprocessing and feature engineering was executed via `pytest`:

```powershell
python -m pytest tests/
```

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-8.3.4, pluggy-1.5.0
rootdir: C:\Users\Dell\Downloads\beijing+multi+site+air+quality+data
plugins: anyio-4.11.0
collected 14 items

tests\test_feature_engineering.py .......                                [ 50%]
tests\test_preprocessing.py .......                                      [100%]

======================== 14 passed, 1 warning in 2.69s ========================
```

All 14 tests passed with zero errors.
