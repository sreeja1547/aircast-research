# AirCast Research: Preprocessing Methodology & Mathematical Formulation

**Document Version:** 1.0.0  
**Module Reference:** `src/preprocessing/` (`cleaner.py`, `splitter.py`, `transformers.py`)  
**Test Suite:** `tests/test_preprocessing.py` (7 tests, 100% passing)  

---

## 1. Problem Formulation & Strict Information Boundaries

Let $\mathcal{S} = \{s_1, s_2, \dots, s_{12}\}$ denote the set of 12 spatial air quality monitoring stations in the Beijing metropolitan network. For each station $s \in \mathcal{S}$ at discrete hourly timestamp $t \in \{1, \dots, T_s\}$, the observational vector is defined as:

$$\mathbf{z}_s(t) = \left[ y_s(t), \mathbf{p}_s(t), \mathbf{m}_s(t) \right]^T$$

where:
- $y_s(t) \in \mathbb{R}^+$ is the primary particulate concentration ($PM_{2.5}$).
- $\mathbf{p}_s(t) = [PM_{10}(t), SO_2(t), NO_2(t), CO(t), O_3(t)]^T$ represents co-pollutant telemetry.
- $\mathbf{m}_s(t) = [TEMP(t), PRES(t), DEWP(t), RAIN(t), WSPM(t), wd(t)]^T$ represents surface meteorological measurements.

The objective of direct multi-horizon forecasting is to estimate the future concentration $y_s(t + h)$ for horizons $h \in \{1, 6, 24\}$ hours ahead:

$$\hat{y}_s(t + h) = f_h\left( \mathcal{H}_s(t) \right)$$

where $\mathcal{H}_s(t)$ represents the historical filtration containing **only information observed at or prior to prediction time $t$**:

$$\mathcal{H}_s(t) = \sigma\left( \{ \mathbf{z}_s(\tau) : \tau \le t \} \right)$$

---

## 2. Target Handling: Absolute Ban on Target Imputation

### Axiom of Target Integrity
In supervised time-series forecasting, synthetically imputing unobserved target values $y_s(t + h)$ introduces severe artifacts:
1. Interpolation methods (e.g., linear, cubic spline) borrow future observations ($\tau > t + h$), leaking information into the target label.
2. Forward-filling targets artificially flattens pollution trajectories and biases variance estimates toward zero.
3. Imputing targets with unconditional means or medians distorts the gradient signals during backpropagation or loss minimization.

### Implemented Rule
Let $y_s(t + h)$ denote the supervised training label for horizon $h$. If $y_s(t + h)$ is unobserved (`NaN`):
$$\text{If } y_s(t + h) = \text{NaN} \implies \text{Drop sample } (\mathcal{H}_s(t), y_s(t + h)) \text{ from training and evaluation sets.}$$

The function `drop_unobserved_targets(df, target_col)` in [src/preprocessing/cleaner.py](../src/preprocessing/cleaner.py) strictly enforces this condition. Feature vectors at time $t$ remain valid for historical lag construction even if the label at $t+h$ is omitted.

---

## 3. Station-Aware Causal Imputation for Historical Inputs

While forecasting targets must never be imputed, missing entries in historical feature inputs $\mathbf{z}_s(\tau)$ for $\tau \le t$ must be resolved to form dense input matrices for machine learning algorithms.

### Mathematical Formulation
To guarantee strict causality and avoid spatial contamination:
1. **Intra-Station Isolation:** Imputation operates strictly within the time series of station $s$, independently of all other stations $s' \neq s$.
2. **Causal Forward Fill:** For any feature $x_{s}(t)$ that is unobserved, the imputed value $\tilde{x}_{s}(t)$ is given by the most recent valid historical measurement:
   $$\tilde{x}_{s}(t) = x_{s}(t^*), \quad \text{where } t^* = \max \{ \tau < t : x_{s}(\tau) \text{ is observed} \}$$
3. **Streak Capping & Training Fallback:** Because 87.2% of missing gaps are $\le 3$ hours, forward fill is highly effective. However, if a gap exceeds $K = 72$ hours, stale historical values are abandoned to avoid propagating obsolete states. Any residual unobserved entry is imputed using the median of that feature computed strictly on the training partition of station $s$:
   $$\tilde{x}_{s}(t) = \text{Median}_{\text{train}}(x_{s}) \quad \text{if } (t - t^*) > 72\text{ hours or no prior observation exists.}$$
4. **Binary Missingness Indicators:** To allow non-linear models to learn that a sensor measurement was unobserved, a binary indicator is appended:
   $$m_{x, s}(t) = \begin{cases} 1 & \text{if } x_s(t) \text{ was originally unobserved} \\ 0 & \text{if } x_s(t) \text{ was observed} \end{cases}$$

This logic is implemented in `StationAwareCausalImputer` in [src/preprocessing/cleaner.py](../src/preprocessing/cleaner.py) and verified in `tests/test_preprocessing.py`.

---

## 4. Chronological Splitting with 24-Hour Boundary Purge

### Leakage Vector in Multi-Horizon Time Series
Consider direct forecasting for horizon $h = 24$ hours. Suppose a conventional split partitions the timeline into training set $[T_0, T_{\text{train}}]$ and validation set $[T_{\text{train}} + 1, T_{\text{val}}]$.
- A training sample constructed at timestamp $t = T_{\text{train}}$ possesses a target label:
  $$y(t + 24) = y(T_{\text{train}} + 24)$$
- This target $y(T_{\text{train}} + 24)$ falls **24 hours into the validation period**!
- During model training, the optimizer observes ground-truth labels belonging to the validation period, compromising validation independence.

### The 24-Hour Boundary Purge (Embargo Buffer)
To eliminate label leakage across splits, AirCast Research enforces an embargo buffer of at least 24 hours between adjacent temporal partitions:

```
[--- Training Partition ---] | [Purge 1] | [--- Validation Partition ---] | [Purge 2] | [--- Test Partition ---]
2013-03-01 to 2015-12-31    | 24 Hours  | 2016-01-02 to 2016-09-30      | 24 Hours  | 2016-10-02 to 2017-02-28
(34 Months, 24,840h/st)     | Embargo   | (9 Months, 6,552h/st)          | Embargo   | (5 Months, 3,600h/st)
```

### Partition Boundary Specification

| Partition | Start Timestamp | End Timestamp | Calendar Duration | Hours per Station | Total Observations (12 Stations) | Purpose |
| :--- | :--- | :--- | :--- | :-: | :-: | :--- |
| **Training** | 2013-03-01 00:00:00 | 2015-12-31 23:00:00 | 34 months (1,035 days) | 24,840 | 298,080 | Model parameter estimation & pipeline fitting |
| **Purge Buffer 1** | 2016-01-01 00:00:00 | 2016-01-01 23:00:00 | 24 hours (1 day) | 24 | 288 | Target embargo buffer (guarantees $y(t+24) \le \text{Train}$) |
| **Validation** | 2016-01-02 00:00:00 | 2016-09-30 23:00:00 | 9 months (273 days) | 6,552 | 78,624 | Hyperparameter tuning, early stopping, model selection |
| **Purge Buffer 2** | 2016-10-01 00:00:00 | 2016-10-01 23:00:00 | 24 hours (1 day) | 24 | 288 | Target embargo buffer (guarantees $y(t+24) \le \text{Val}$) |
| **Final Test** | 2016-10-02 00:00:00 | 2017-02-28 23:00:00 | 5 months (150 days) | 3,600 | 43,200 | Final untouched out-of-sample evaluation |

Implemented in `TemporalPurgedSplitter` in [src/preprocessing/splitter.py](../src/preprocessing/splitter.py).

---

## 5. Spatial Partitioning: Leave-One-Station-Out (LOSO) Protocol

To evaluate cross-station generalisation (Research Question 4 and Experiment 7), a separate spatial partitioning protocol is enforced:

$$\forall s \in \mathcal{S}: \quad \mathcal{D}_{\text{train}}^{(-s)} = \bigcup_{s' \in \mathcal{S} \setminus \{s\}} \mathcal{D}_{s'}, \quad \mathcal{D}_{\text{test}}^{(s)} = \mathcal{D}_s$$

### Strict Spatial Isolation Protocol
1. For each of the 12 evaluation rounds, station $s$ is held out completely as the unseen target domain.
2. The remaining 11 stations form the development set.
3. Preprocessing transformers, causal imputers, categorical encodings, and scalers are fitted **exclusively on $\mathcal{D}_{\text{train}}^{(-s)}$**.
4. Station $s$ data is transformed using the parameters learned from the 11 stations. No target statistics from station $s$ enter the training pipeline.
5. Implemented in `StationHoldoutSplitter` in [src/preprocessing/splitter.py](../src/preprocessing/splitter.py).

---

## 6. Physical Feature Transformations

### A. Meteorological Wind Vector Decomposition
Wind direction is reported as a 16-point compass heading $\theta_{\text{compass}} \in [0^\circ, 360^\circ)$. In atmospheric physics, wind is a vector quantity with magnitude equal to wind speed ($WSPM$) and orientation pointing in the direction towards which the wind is blowing.

To avoid artificial mathematical discontinuities at the North boundary ($359^\circ \to 0^\circ$), wind is decomposed into orthogonal Cartesian components:
- **Zonal Component ($u$):** West-to-East wind velocity:
  $$u = -WSPM \cdot \sin\left(\frac{2\pi \cdot \theta_{\text{deg}}}{360}\right)$$
- **Meridional Component ($v$):** South-to-North wind velocity:
  $$v = -WSPM \cdot \cos\left(\frac{2\pi \cdot \theta_{\text{deg}}}{360}\right)$$
- Circular direction components:
  $$\text{wd}_{\sin} = \sin\left(\frac{2\pi \cdot \theta_{\text{deg}}}{360}\right), \quad \text{wd}_{\cos} = \cos\left(\frac{2\pi \cdot \theta_{\text{deg}}}{360}\right)$$

### B. Periodic Temporal Encodings
Diurnal and seasonal dynamics are mapped to the 2D unit circle:
- **Diurnal Cycle:**
  $$\text{hour}_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour}_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
- **Annual Seasonal Cycle:**
  $$\text{month}_{\sin} = \sin\left(\frac{2\pi \cdot (\text{month} - 1)}{12}\right), \quad \text{month}_{\cos} = \cos\left(\frac{2\pi \cdot (\text{month} - 1)}{12}\right)$$

### C. Moisture Proxy: Dew Point Depression
Relative humidity is derived physically from ambient temperature ($TEMP$) and dew point temperature ($DEWP$):
$$\Delta_{\text{dew}} = TEMP - DEWP \quad (^\circ\text{C})$$
When $\Delta_{\text{dew}} \to 0^\circ\text{C}$, the atmosphere approaches saturation ($RH \to 100\%$), triggering hygroscopic growth of secondary particulate matter.

### D. Robust Feature Scaling
Because extreme pollution events reach values over $10\times$ the median, traditional standard scaling (zero mean, unit variance) is heavily skewed by extreme outliers. AirCast Research provides `RobustFeatureScaler`:

$$\tilde{x} = \frac{x - \text{Median}_{\text{train}}(x)}{\text{IQR}_{\text{train}}(x)}, \quad \text{where } \text{IQR} = Q_3(x) - Q_1(x)$$

The median and IQR are computed strictly on the training partition during `fit()`.

---

## 7. Leakage Prevention Matrix & Verification Status

| Potential Leakage Vector | Specific Risk | AirCast Prevention Mechanism | Verification Unit Test |
| :--- | :--- | :--- | :--- |
| **Future Target Peeking** | Imputing missing labels with future interpolation | Strict sample dropping (`drop_unobserved_targets`); targets are never imputed | `test_drop_unobserved_targets` |
| **Historical Input Lookahead** | Backward-filling inputs using future sensor values | Strictly causal forward-fill (`StationAwareCausalImputer`) | `test_causal_imputer_no_lookahead` |
| **Multi-Horizon Boundary Spill** | Forecast horizon $h=24$ target spilling into validation/test | 24-hour embargo buffer (`TemporalPurgedSplitter`) | `test_temporal_purged_splitter` |
| **Spatial Contamination** | Fitting imputers/scalers on held-out station data | Strict domain isolation (`StationHoldoutSplitter`) | `test_station_holdout_splitter` |
| **Distributional Peeking** | Computing scaler mean/std or imputer medians across full series | `fit()` executed strictly on training split; `transform()` applies learned parameters | `test_scaler_fitted_on_train_only` |
| **Circular Discontinuity** | Modulo boundary jumps ($23\text{h} \to 0\text{h}$, $359^\circ \to 0^\circ$) | Trigonometric decomposition ($\sin / \cos$) | `test_cyclical_transformer`, `test_wind_vector_transformer` |

---

## 8. Automated Test Execution Results

The automated preprocessing test suite was executed in the target environment:

```powershell
pytest tests/test_preprocessing.py
```

### Execution Log
```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-8.3.4, pluggy-1.5.0
rootdir: C:\Users\Dell\Downloads\beijing+multi+site+air+quality+data
plugins: anyio-4.11.0
collected 7 items

tests\test_preprocessing.py .......                                      [100%]

======================== 7 passed, 1 warning in 2.15s =========================
```

All 7 test cases passed with zero failures.
