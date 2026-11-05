# AirCast Research: Phase 2 Final Technical & Methodological Verification

**Document Version:** 1.0.0  
**Verification Date:** November 2026  
**Auditor:** Senior Applied Scientist & Machine Learning Architect  
**Objective:** Independent technical review and verification of Preprocessing and EDA implementations prior to Phase 3.

---

## 1. Reproducibility of Statistical Metrics

All empirical values presented in [research/eda_report.md](eda_report.md) were independently reconciled against [artifacts/metrics/eda_statistical_summary.json](../artifacts/metrics/eda_statistical_summary.json) and computed directly from the 12 raw station CSV files without modification.

### Verified Metric Alignment

| Quantity | Value in `eda_report.md` | Value in `eda_statistical_summary.json` | Verification Status | Source Dataset Cross-Check |
| :--- | :-: | :-: | :-: | :--- |
| Total observations | 420,768 | 420,768 | Verified Exact | $12\ \text{stations} \times 35,064\ \text{rows} = 420,768$ |
| $PM_{2.5}$ Valid Count | 412,029 | 412,029 | Verified Exact | $420,768 - 8,739\ (\text{unobserved}) = 412,029$ |
| $PM_{2.5}$ Mean | 79.79 $\mu\text{g}/\text{m}^3$ | 79.79 $\mu\text{g}/\text{m}^3$ | Verified Exact | Sample mean across all valid records |
| $PM_{2.5}$ Median | 55.00 $\mu\text{g}/\text{m}^3$ | 55.00 $\mu\text{g}/\text{m}^3$ | Verified Exact | 50th percentile of valid records |
| $PM_{2.5}$ IQR | 91.00 $\mu\text{g}/\text{m}^3$ | 91.00 $\mu\text{g}/\text{m}^3$ | Verified Exact | $Q_3 (111.00) - Q_1 (20.00) = 91.00$ |
| $PM_{2.5}$ Skewness | +2.014 | +2.014 | Verified Exact | Fisher-Pearson standardized moment |
| $PM_{2.5}$ Excess Kurtosis | +5.965 | +5.965 | Verified Exact | Excess kurtosis ($\mu_4 / \sigma^4 - 3$) |
| Heating Season Mean | 94.88 $\mu\text{g}/\text{m}^3$ | 94.88 $\mu\text{g}/\text{m}^3$ | Verified Exact | Filtered Nov 15 – Mar 15 across all years |
| Non-Heating Season Mean | 69.05 $\mu\text{g}/\text{m}^3$ | 69.05 $\mu\text{g}/\text{m}^3$ | Verified Exact | Filtered Mar 16 – Nov 14 across all years |
| Weekday Mean | 78.08 $\mu\text{g}/\text{m}^3$ | 78.08 $\mu\text{g}/\text{m}^3$ | Verified Exact | Monday through Friday observations |
| Weekend Mean | 84.04 $\mu\text{g}/\text{m}^3$ | 84.04 $\mu\text{g}/\text{m}^3$ | Verified Exact | Saturday and Sunday observations |
| $PM_{2.5} > PM_{10}$ Pairs | 21,312 | 21,312 | Verified Exact | Simultaneous valid pairs: **411,813** ($5.175164\%$) |
| Missing $PM_{2.5}$ Streaks $\le 3$h | 87.2% | 87.2% | Verified Exact | Cumulative run-length frequency of unobserved blocks |

---

## 2. Epistemological and Causal Boundaries

Observational correlations between telemetry signals must not be conflated with proven causal mechanisms. In all research documentation and publications:

1. **Planetary Boundary Layer (PBL) Dynamics:**
   - The observed afternoon drop in $PM_{2.5}$ (14:00–16:00) and nocturnal peak (21:00–02:00) are **literature-supported physical hypotheses** consistent with established boundary layer meteorology (e.g., Stull, 1988; Zhang et al., 2017).
   - They represent plausible physical interpretations, not causal relationships proven by this statistical dataset, as vertical lidar or radiosonde profiling was not recorded at these surface stations.

2. **Traffic and Freight Emissions:**
   - Nocturnal spikes matching municipal heavy-duty diesel freight entry windows (post-20:00) are observational associations corroborated by municipal transport reports, not causal proofs.

3. **Regional Industrial Transport vs. Local Primary Emissions:**
   - Southerly wind associations with elevated $PM_{2.5}$ are consistent with synoptic transport from the industrial Hebei plain, as documented in regional trajectory studies (Sun et al., 2014). They are reported as directional meteorological dependencies, not direct chemical tracer apportionments.

4. **Secondary Inorganic Aerosol (SIA) Formation:**
   - The observed negative correlation between dew point depression ($TEMP - DEWP$) and $PM_{2.5}$ ($\rho = -0.459$) aligns with literature on aqueous-phase sulfate/nitrate synthesis under high humidity. It is treated as a predictive physical proxy, not direct in-situ chemical kinetics.

---

## 3. StationAwareCausalImputer Under Leave-One-Station-Out (LOSO) Isolation

### Code Audit of Imputation Under Spatial Quarantine
In [src/preprocessing/cleaner.py](../src/preprocessing/cleaner.py):
```python
st_medians = self.station_medians_.get(station, self.global_medians_)
fallback = st_medians.get(col, self.global_medians_.get(col, 0.0))
```
### Verification Findings:
1. When evaluating a held-out station $s_{\text{test}} \notin \mathcal{D}_{\text{train}}$, the imputer's dictionary `self.station_medians_` contains **only the 11 training stations**.
2. Calling `self.station_medians_.get(s_{\text{test}}, self.global_medians_)` returns `self.global_medians_`.
3. `self.global_medians_` was computed strictly across $\mathcal{D}_{\text{train}}$ during `fit()`.
4. Therefore, **zero statistics, observations, or moments from the held-out station enter the fallback imputation pipeline**. Complete spatial quarantine is maintained.

---

## 4. Multi-Horizon Target Alignment and Missing Target Handling

In multi-horizon direct forecasting, separate models are trained for horizons $h \in \{1, 6, 24\}$ hours:
$$y_s^{(h)}(t) = PM_{2.5, s}(t + h)$$

### Verification Rules:
1. **Station Independence:** Target shifting ($t + h$) is applied strictly within each station time series. A timestamp shift at station $A$ can never access records from station $B$.
2. **Missing Target Exclusion:** If $PM_{2.5, s}(t + h)$ is unobserved (`NaN`), the sample $(\mathbf{x}_s(t), y_s^{(h)}(t))$ is excluded from training and evaluation. Target labels are **never synthetically imputed**.
3. **Feature-Target Chronology:** For a prediction made at time $t$, feature vector $\mathbf{x}_s(t)$ contains only historical measurements $\tau \le t$, while target $y_s^{(h)}(t)$ is strictly at $t + h$.

---

## 5. Verification of the 24-Hour Temporal Boundary Purge

The temporal validation protocol partitions the dataset into three chronological intervals:
- **Training Set:** `2013-03-01 00:00:00` to `2015-12-31 23:00:00`
- **Purge Buffer 1:** `2016-01-01 00:00:00` to `2016-01-01 23:00:00` (24 hours)
- **Validation Set:** `2016-01-02 00:00:00` to `2016-09-30 23:00:00`
- **Purge Buffer 2:** `2016-10-01 00:00:00` to `2016-10-01 23:00:00` (24 hours)
- **Final Test Set:** `2016-10-02 00:00:00` to `2017-02-28 23:00:00`

### Mathematical Proof of Purge Completeness:
- For the longest forecasting horizon $h = 24$, the latest possible training sample is constructed at $t_{\text{max, train}} = \text{2015-12-31 23:00:00}$.
- Its target label corresponds to:
  $$t_{\text{max, train}} + 24\text{h} = \text{2016-01-01 23:00:00}$$
- The validation set begins at $t_{\text{min, val}} = \text{2016-01-02 00:00:00}$.
- Difference: $t_{\text{min, val}} - (t_{\text{max, train}} + 24\text{h}) = 1\ \text{hour} > 0$.
- **Conclusion:** Not a single training target reaches into the validation set, and not a single validation target reaches into the test set.

---

## 6. Audit of the Eight EDA Visualizations

All 8 figures in [artifacts/plots/](../artifacts/plots) were audited for scientific publishing standards:

1. `eda_pollutant_distributions.png`: Units clearly specified ($\mu\text{g}/\text{m}^3$); median vs. mean divergence marked; sample sizes indicated.
2. `eda_temporal_seasonal_patterns.png`: Diurnal hours formatted in 24-hour Local Beijing Time (UTC+8); heating season explicitly defined as Nov 15–Mar 15.
3. `eda_station_comparison.png`: Stations sorted monotonically by median pollution; outlier suppression applied only for visual box clarity while summary statistics retain full distributions.
4. `eda_meteorological_relationships.png`: Physical threshold ($2.0\ \text{m}/\text{s}$) annotated; dew point depression defined as $TEMP - DEWP$; wind rose shows directional bimodality.
5. `eda_correlation_matrix.png`: Separate Pearson and Spearman matrices; colorbar normalized to $[-1, +1]$; annotations rounded to 2 decimal places.
6. `eda_missingness_patterns.png`: Unobserved rate heatmap shows exact station-by-pollutant percentages; streak length histogram uses log/linear scaling with median marked.
7. `eda_pm_ratio_discrepancy.png`: Exact denominator ($N = 411,813$) documented in title and axis; 1:1 line shown in red; station breakdown plotted as percentages.
8. `eda_extreme_pollution_events.png`: Standard AQI breakpoints displayed; historic Red Alert episode shown with station range envelope.

---

## 7. Air Quality Index (AQI) Breakpoint Standards

To ensure clarity, the air quality categories used in AirCast Research are grounded in established regulatory standards:

### Standard Comparison: US EPA vs. China MEP (HJ 633-2012)

| Category Descriptor | US EPA NAAQS 24-h Breakpoint ($\mu\text{g}/\text{m}^3$) | China MEP (HJ 633-2012) Daily Breakpoint ($\mu\text{g}/\text{m}^3$) | AirCast Usage |
| :--- | :---: | :---: | :--- |
| **Good / Excellent** | $0.0 - 12.0$ | $0.0 - 35.0$ | Documented under both; EPA primary in EDA ($15.24\%$) |
| **Moderate / Good** | $12.1 - 35.4$ | $35.1 - 75.0$ | EPA Moderate ($22.11\%$) |
| **Unhealthy for Sensitive / Lightly Polluted**| $35.5 - 55.4$ | $75.1 - 115.0$ | EPA Sensitive ($12.92\%$) |
| **Unhealthy / Moderately Polluted** | $55.5 - 150.4$ | $115.1 - 150.0$ | EPA Unhealthy ($34.52\%$) |
| **Very Unhealthy / Heavily Polluted** | $150.5 - 250.4$ | $150.1 - 250.0$ | EPA Very Unhealthy ($10.74\%$) |
| **Hazardous / Severely Polluted** | $> 250.4$ | $> 250.0$ | Both converge at $\approx 250\ \mu\text{g}/\text{m}^3$ ($4.47\%$) |

Both standards converge closely at the critical severe pollution threshold ($250\ \mu\text{g}/\text{m}^3$), where emergency public health warnings are mandated.

---

## 8. Guarantee: Zero Future Weather Telemetry as Model Inputs

In operational deployment, true future weather observations at time $t + h$ are unavailable.
- Under direct forecasting:
  $$\hat{y}(t + h) = f_h(\mathbf{x}(t))$$
- $\mathbf{x}(t)$ contains **only surface weather observed at or before $t$**:
  $$TEMP(t), PRES(t), DEWP(t), RAIN(t), WSPM(t), \mathbf{w}(t)$$
  and their historical rolling/lagged statistics.
- **Zero actual future weather values ($TEMP(t+h), WSPM(t+h)$) are provided as predictors.**

---

## 9. Test Suite Verification

The preprocessing test suite (`tests/test_preprocessing.py`) passes 100% of cases:
- `test_causal_imputer_no_lookahead` (Passed)
- `test_drop_unobserved_targets` (Passed)
- `test_temporal_purged_splitter` (Passed)
- `test_station_holdout_splitter` (Passed)
- `test_cyclical_transformer` (Passed)
- `test_wind_vector_transformer` (Passed)
- `test_scaler_fitted_on_train_only` (Passed)

---

## 10. Identified Engineering and Scientific Limitations

1. **Station Spatial Coordinates:** The raw dataset provides station names without embedded latitude/longitude columns. Spatial relationships are currently modeled via categorical station indicators and station-specific historical statistics.
2. **Atmospheric Boundary Layer Height (PBLH):** PBLH is not directly measured in the surface dataset. Diurnal and seasonal cyclical features serve as proxy variables for PBL convective lifting and nocturnal inversions.
3. **Precipitation Sparsity:** 95.98% of rainfall observations are $0.0\ \text{mm}$. Rain must be modeled as an episodic binary washout indicator alongside continuous accumulation.

---

## Verification Verdict

**Technical Review Status:** **APPROVED**. All preprocessing components, mathematical formulations, causal boundary safeguards, and leakage-prevention mechanisms are verified. The codebase is cleared to proceed to Phase 3.
