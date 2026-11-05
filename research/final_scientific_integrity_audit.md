# AirCast Research — Final Scientific Integrity and Release Audit

**Auditor:** Independent Senior Applied Scientist, Research Reviewer, ML Systems Architect & Statistical Auditor  
**Project:** AirCast Research: A Research-Driven Framework for Explainable, Uncertainty-Aware and Generalizable Multi-Horizon Air Quality Forecasting  
**Dataset:** Beijing Multi-Site Telemetry Network (PRSA 2017; 420,768 records, 12 monitoring stations, 2013–2017)  
**Audit Date:** November 5, 2026  
**Audit Scope:** Source code, experimental configurations, saved model binaries, JSON metric artifacts, evaluation scripts, API/Dashboard serving layers, test suites, and technical research reports across Phases 1 through 10.  

---

## 1. Executive Assessment

This audit provides an independent, rigorous, and critical review of the entire AirCast Research codebase, experimental pipeline, empirical artifacts, and serving infrastructure.

### Overall Judgment
The AirCast Research framework is a **technically sophisticated, methodologically rigorous, and highly competent applied machine learning research system**. The project demonstrates excellent software architecture, strictly enforced chronological data partitioning, zero target imputation, causal feature engineering, and comprehensive benchmarking across multiple taxonomic paradigms.

However, an independent scientific audit must separate verified empirical findings from methodological limitations, heuristic compromises, and domain overclaims:
1. **Conformal Coverage Non-Stationarity:** Split conformal prediction does **not** maintain 90% test coverage under real-world winter heating conditions. Due to severe temporal distribution shift between the 2016 validation period (Jan–Sep) and the 2016–2017 winter test period (Oct–Feb), empirical test coverage drops from the nominal 90.0% to **83.35% at $h=1\text{h}$**, **79.59% at $h=6\text{h}$**, and **71.80% at $h=24\text{h}$**, with extreme coverage dropping to **11.85%** during hazardous pollution episodes ($>250\ \mu\text{g}/\text{m}^3$).
2. **API Conformal Serving Heuristic:** While the core package contains an exact non-parametric `ConformalPredictor`, the production API service in `api/services.py` currently computes prediction intervals using a hardcoded heuristic margin lookup table (`margin_map = {1: 15.0, 6: 35.0, 24: 55.0}`) scaled by a Gaussian z-factor approximation, rather than dynamically querying the calibrated empirical quantiles.
3. **LOSO Inference Clarification:** Leave-One-Station-Out (LOSO) generalisation is **not** spatial field interpolation across unmonitored geographic coordinates (kriging). At inference time, the model utilizes the held-out station's own local sensor readings, co-pollutants, and historical autoregressive lags. It evaluates model parameter transferability to a new sensor installation, not zero-sensor spatial estimation.
4. **Station Identifier Feature in LOSO:** In `src/models/base.py`, the categorical `station` column is mapped to integer floats ($0.0 \dots 10.0$) during training. For an unseen held-out station in LOSO, it is assigned `-1.0`. In tree ensembles, `-1.0` is treated as a numerical value taking the lowest branch, which is an artificial numerical extrapolation.
5. **Observational vs. Causal Language:** Prior documentation framed TreeSHAP attributions and seasonal pollution surges as causal proof of atmospheric transport and coal heating. In observational telemetry, these represent statistical associations and model decision dependencies, not Pearl-style interventional causal proofs.
6. **Incomplete Annual Cycle in Test Partition:** The test set spans October 2, 2016 to February 28, 2017 (5 months). It contains zero spring or summer observations.

### Summary Classification Matrix

| Audit Dimension | Target Task | Status Verdict | Summary of Findings |
| :--- | :---: | :---: | :--- |
| **Data Partitioning & Causality** | Task 1 | **VERIFIED** | Chronological splits ($\le 2015$, 2016, 2016–17) with strict 24h embargo; zero future weather leakage. |
| **Target & Imputation Integrity** | Task 1 | **VERIFIED** | Targets shifted from raw unimputed PM2.5; missing targets dropped; imputer fitted on train only. |
| **Multi-Horizon Benchmarking** | Task 1 | **VERIFIED** | All 7 architectures evaluated on identical test samples per horizon; MAE vs RMSE winners distinguished. |
| **LSTM Architecture & Boundaries**| Task 1 | **VERIFIED WITH LIMITATIONS** | Tabular sequence formulation (`seq_len=1`); operates on engineered features without sequence bleeding. |
| **LOSO Generalisation Protocol** | Task 2 | **VERIFIED WITH LIMITATIONS** | Evaluates transfer using local telemetry at inference; unseen station mapped to `-1.0`; macro-averaged. |
| **Conformal Coverage Validity** | Task 3 | **REQUIRES CORRECTION** | Nominal 90% violated on test split ($71.8\%$ at $24\text{h}$) due to seasonal distribution shift. |
| **TreeSHAP Explainability** | Task 4 | **VERIFIED WITH LIMITATIONS** | Exact C++ TreeSHAP verified; wind signs match meteorology; re-framed from causal to associative. |
| **Robustness & Stress Testing** | Task 5 | **VERIFIED WITH LIMITATIONS** | Post-engineered feature noise without raw propagation; single seed; wind sensor loss is critical. |
| **Temporal & Seasonal Analysis** | Task 6 | **VERIFIED WITH LIMITATIONS** | Test split covers 5 months (Oct–Feb); associations observed, not causal heating proof. |
| **API & Dashboard Serving** | Task 7 | **VERIFIED WITH LIMITATIONS** | Local research prototype; API uses heuristic margin map rather than serialized conformal quantiles. |
| **Automated Test Suite** | Task 8 | **VERIFIED** | 43/43 unit tests passing (100% pass rate in 34.7s). |

---

## 2. Verified Findings

The following foundational components were inspected at the source code level and verified to be mathematically and scientifically correct:
- **Strict Chronological Embargo:** `TemporalPurgedSplitter` in `src/preprocessing/splitter.py` enforces exact timestamp boundaries:
  - Training: `2013-03-01 00:00:00` to `2015-12-31 23:00:00` (291,574 samples)
  - Purge Buffer 1: `2016-01-01 00:00:00` to `2016-01-01 23:00:00` (24-hour embargo)
  - Validation: `2016-01-02 00:00:00` to `2016-09-30 23:00:00` (77,316 samples)
  - Purge Buffer 2: `2016-10-01 00:00:00` to `2016-10-01 23:00:00` (24-hour embargo)
  - Test: `2016-10-02 00:00:00` to `2017-02-28 23:00:00` (42,278 to 42,554 samples depending on $h$)
- **Zero Target Imputation:** In `MultiHorizonForecastingDataset.prepare_dataset()` (`src/data/forecasting_dataset.py`), the target column is copied from raw $PM_{2.5}$ *before* causal imputation:
  ```python
  raw_target_col = f"{self.target_col}_raw_unimputed"
  df_work[raw_target_col] = df_work[self.target_col]
  ```
  Forward shifts (`shift(-h)`) are executed strictly within station groups. Samples with missing forward targets are dropped via `df[target_col].notnull()`. Target imputation is strictly zero.
- **Causal Feature Extraction:** In `StationAwareFeatureBuilder` (`src/features/builder.py`), all autoregressive lags (`shift(lag)`) and rolling statistics (`rolling(window=w, min_periods=1)`) are partitioned strictly by station. At prediction timestamp $t$, only telemetry observed at or before $t$ is included. Weather features ($TEMP, PRES, DEWP, WSPM$) are instantaneous surface measurements at time $t$; no future weather forecasts or leakages are present.
- **Training-Only Feature Conditioning:**
  - `StationAwareCausalImputer` is fitted strictly on the training partition (`dt <= splitter.train_end`) in `scripts/prepare_processed_data.py`.
  - Feature scalers (`RobustScaler` in Linear/Ridge, empirical standardizers in LSTM) are fitted strictly on `X_train` inside their respective `fit()` methods.
- **Standardized Evaluation Metrics:** `calculate_regression_metrics()` in `src/evaluation/metrics.py` calculates MAE, RMSE, and $R^2$ independently using `sklearn.metrics`. The edge case of zero variance is explicitly handled.
- **Automated Test Suite:** Full test suite execution (`pytest tests/`) passes 43 out of 43 tests (100% pass rate in 34.73s).

---

## 3. Problems Identified

The audit identified six specific methodological, operational, or conceptual issues:

### Problem 1: Conformal Prediction Breakdown Under Seasonal Distribution Shift
- **Defect:** Prior documentation claimed conformal prediction guarantees $\ge 90\%$ marginal coverage on test data ($91.4\%$).
- **Audit Reality:** Split conformal prediction guarantees finite-sample coverage **only under data exchangeability**. In atmospheric time series, severe winter stagnation introduces non-stationarity.
- **Empirical Measurement:** When calibrated on the 2016 validation partition (Jan–Sep 2016) and evaluated on the held-out winter test set (Oct 2016–Feb 2017):
  - At $h=1\text{h}$: Empirical coverage is **$83.35\%$** (below nominal 90%).
  - At $h=6\text{h}$: Empirical coverage is **$79.59\%$** (below nominal 90%).
  - At $h=24\text{h}$: Empirical coverage is **$71.80\%$** (severely under-covering by $-18.2\%$).
  - For hazardous episodes ($PM_{2.5} > 250\ \mu\text{g}/\text{m}^3$): Empirical coverage collapses to **$11.85\%$**.

### Problem 2: API Conformal Serving Approximation
- **Defect:** In `api/services.py` (lines 91–99), prediction intervals are computed via:
  ```python
  margin_map = {1: 15.0, 6: 35.0, 24: 55.0}
  base_margin = margin_map.get(horizon, 45.0)
  z_factor = 1.645 if nominal_coverage <= 0.90 else 1.96
  margin = base_margin * (z_factor / 1.645)
  ```
  This is a static heuristic margin lookup with a Gaussian multiplier, not an invocation of the calibrated `ConformalPredictor` quantiles.

### Problem 3: Station Identifier Encoding in LOSO Evaluation
- **Defect:** In `src/models/base.py`, string categorical columns (`station`) are encoded as integer floats ($0.0 \dots 10.0$) on training data. In LOSO cross-station evaluation, the held-out station is mapped to `-1.0`.
- **Audit Reality:** Decision trees branch on `-1.0` as an ordinal value smaller than all training stations. While station ID has low feature importance compared to wind vectors and lags, encoding nominal unseen stations as `-1.0` is an ad-hoc numerical artifact.

### Problem 4: Robustness Perturbation Methodology
- **Defect:** In `src/evaluation/robustness.py`, Gaussian noise is injected directly into post-engineered feature columns in `X_test` (including circular harmonics like `hour_sin` and binary indicators like `is_weekend`) rather than perturbing raw sensor measurements and propagating them causally through the feature pipeline.
- **Audit Reality:** The test was run on a single fixed random seed (`np.random.seed(42)`). Because tree ensembles average hundreds of split thresholds over 160 trees and perturbations were zero-mean, perturbations largely canceled out, producing an artificially low reported degradation ($+0.26\%$).

### Problem 5: Causal Overclaiming
- **Defect:** Atmospheric interpretations in Phase 2–7 documentation claimed that models "proved" wind ventilation thresholds and chemical secondary aerosol formation.
- **Audit Reality:** Observational machine learning models identify predictive correlations and split dependencies, not physical causality.

### Problem 6: Test Partition Annual Cycle Incompleteness
- **Defect:** The test partition spans only October 2, 2016 through February 28, 2017 (5 months).
- **Audit Reality:** The evaluation contains no observations from spring (March–May) or summer (June–August). Seasonal comparisons compare October (autumn) against November–February (winter), conflating heating emissions with natural meteorological winter boundary layer dynamics.

---

## 4. Corrections Made

To address these audit findings, the following actions and calibrations were completed:
1. **Stratified Conformal Audit:** Executed `scripts/audit_scientific_integrity.py` and exported `artifacts/metrics/conformal_stratified_audit.json` documenting the true empirical coverage across all 3 horizons, all 12 stations, heating vs. non-heating seasons, and 5 AQI concentration tiers.
2. **LOSO Persistence Baseline Reconciliation:** Executed explicit station-by-station persistence baseline evaluations across all 12 held-out stations, exporting `artifacts/metrics/loso_persistence_audit.json`.
3. **Scientific Language Recalibration:** Re-framed all atmospheric explanations from causal assertions to observational and model-association findings.
4. **Local Prototype Demarcation:** Formally classified the FastAPI and Streamlit deployments as local prototype research demonstrations, disclaiming any unverified cloud scalability or enterprise production claims.

---

## 5. Benchmark Integrity (Task 1 Audit)

### Evaluation Integrity Verification
- **Sample Uniformity:** For a given horizon $h$, all seven models evaluate on the exact same sample count ($42,554$ at $h=1\text{h}$; $42,494$ at $h=6\text{h}$; $42,278$ at $h=24\text{h}$).
- **Target Inversion:** Targets ($y$) were never normalized or log-transformed during dataset construction; models predict in raw $\mu\text{g}/\text{m}^3$ physical units, clamped at $\ge 0.0$. No inverse scaling was omitted.
- **LSTM Sequence Formulation:** `LSTMForecaster` in `src/models/deep_learning.py` converts the 2D tabular feature matrix $(N, 47)$ into a 3D tensor of shape $(N, 1, 47)$ (`seq_len = 1`). Because sequence length is 1, it treats the engineered 47 multimodal features as an instant state vector. This design prevents accidental cross-station boundary bleeding, but means the recurrent cells do not operate on an unrolled multi-step historical sequence.

### Quantitative Benchmark Matrix & Winner Reconciliation

| Horizon ($h$) | Architecture | Test MAE ($\mu\text{g}/\text{m}^3$) | Test RMSE ($\mu\text{g}/\text{m}^3$) | Test $R^2$ | Extreme MAE ($y \ge 150$) | Normal MAE ($y < 150$) | Pearson $r$ | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h=1\text{h}$** | Persistence | 12.42 | 24.73 | 0.9437 | 21.68 | 9.39 | 0.9718 | Baseline |
| | Linear Regression | 11.96 | 23.19 | 0.9505 | 20.53 | 9.16 | 0.9750 | Scaled |
| | Ridge Regression | 11.96 | 23.19 | 0.9505 | 20.53 | 9.16 | 0.9749 | Scaled |
| | HistGradientBoosting| 11.68 | 23.12 | 0.9508 | 21.84 | 8.36 | 0.9751 | Tree |
| | **Random Forest** | **11.52** | 23.08 | 0.9509 | 21.08 | 8.40 | 0.9752 | **BEST MAE** |
| | XGBoost | 11.57 | 23.33 | 0.9499 | 21.75 | 8.25 | 0.9746 | Tree |
| | **PyTorch LSTM** | 11.57 | **22.29** | **0.9542** | 21.74 | **8.25** | **0.9772** | **BEST RMSE & $R^2$** |
| **$h=6\text{h}$** | Persistence | 41.50 | 70.70 | 0.5398 | 69.72 | 32.33 | 0.7701 | Baseline |
| | Linear Regression | 39.24 | 63.36 | 0.6303 | 69.18 | 29.51 | 0.7948 | Scaled |
| | Ridge Regression | 39.24 | 63.37 | 0.6303 | 69.19 | 29.51 | 0.7948 | Scaled |
| | HistGradientBoosting| 37.26 | 63.47 | 0.6291 | 69.27 | 26.86 | 0.7934 | Tree |
| | Random Forest | 37.68 | 63.53 | 0.6284 | 68.61 | 27.63 | 0.7928 | Tree |
| | **XGBoost** | **36.95** | 62.57 | 0.6395 | 68.58 | **26.68** | 0.8001 | **BEST MAE** |
| | **PyTorch LSTM** | 37.22 | **61.84** | **0.6479** | **68.03** | 27.21 | **0.8052** | **BEST RMSE & $R^2$** |
| **$h=24\text{h}$** | Persistence | 78.32 | 113.32 | -0.1806 | 114.49 | 66.78 | 0.4120 | Failed Skill |
| | **Linear Regression** | 66.79 | **91.40** | **0.2319** | 118.12 | **50.41** | **0.4881** | **BEST RMSE & $R^2$** |
| | Ridge Regression | 66.79 | **91.40** | **0.2319** | 118.14 | **50.40** | 0.4882 | **BEST RMSE & $R^2$** |
| | HistGradientBoosting| 67.76 | 95.00 | 0.1703 | **113.67** | 53.11 | 0.4268 | Tree |
| | Random Forest | 68.83 | 96.64 | 0.1413 | 115.67 | 53.88 | 0.4060 | Tree |
| | XGBoost (Default) | 67.26 | 93.41 | 0.1978 | 119.40 | 50.61 | 0.4451 | Tree |
| | XGBoost (Tuned Exp 4)| 66.29 | 92.15 | 0.2193 | 119.09 | 49.43 | 0.4722 | Tuned Tree |
| | **PyTorch LSTM** | **66.47** | 92.48 | 0.2137 | 114.33 | 51.19 | 0.4627 | **BEST MAE** |

### Explicit Distinction Between Best MAE and Best RMSE:
- **At $h=1\text{h}$:** Random Forest achieves the lowest MAE ($11.52\ \mu\text{g}/\text{m}^3$), while PyTorch LSTM achieves the lowest RMSE ($22.29\ \mu\text{g}/\text{m}^3$) and highest $R^2$ ($0.9542$).
- **At $h=6\text{h}$:** XGBoost achieves the lowest MAE ($36.95\ \mu\text{g}/\text{m}^3$), while PyTorch LSTM achieves the lowest RMSE ($61.84\ \mu\text{g}/\text{m}^3$) and highest $R^2$ ($0.6479$).
- **At $h=24\text{h}$:** PyTorch LSTM achieves the lowest default MAE ($66.47\ \mu\text{g}/\text{m}^3$), while Linear/Ridge Regression achieves the lowest RMSE ($91.40\ \mu\text{g}/\text{m}^3$) and highest $R^2$ ($0.2319$). After hyperparameter tuning, XGBoost reaches $\text{MAE} = 66.29\ \mu\text{g}/\text{m}^3$ and $\text{RMSE} = 92.15\ \mu\text{g}/\text{m}^3$.

---

## 6. LOSO Scientific Audit (Task 2 Audit)

### Evaluation Protocol Clarification
- **Zero-Shot Transfer of Model Weights:** For each held-out station, the model is trained strictly on the other 11 stations. No parameter gradient steps or tree splits are trained on the held-out station.
- **Inference Telemetry:** At inference time, the model is evaluated on the held-out station's data using that station's own local sensor readings (particulates, gases, meteorology, and autoregressive lags $\le t$).
- **Distinction from Spatial Interpolation:** LOSO evaluates whether a pre-trained model can be deployed at a **new monitoring station equipped with sensors**, without needing historical model fine-tuning. It is **not** spatial kriging or virtual sensing for an unmonitored location.

### Fold-by-Fold Reconciled Metrics ($h=24\text{h}$)

All 12 folds evaluate XGBoost on the `combined_all` feature group across the entire four-year record of the held-out station:

| Held-Out Station | Domain / Typology | Training Stations | Test Samples | XGBoost MAE | XGBoost RMSE | XGBoost $R^2$ | Persistence MAE | Persistence RMSE | Persistence $R^2$ | Error Reduction vs Persistence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aotizhongxin** | Urban Core | Other 11 | 34,067 | 45.85 | 63.61 | 0.4009 | 60.44 | 90.68 | -0.2176 | $-14.59\ \mu\text{g}/\text{m}^3$ ($-24.1\%$) |
| **Changping** | Suburban | Other 11 | 34,218 | 43.83 | 59.90 | 0.3148 | 53.26 | 79.28 | -0.2001 | $-9.43\ \mu\text{g}/\text{m}^3$ ($-17.7\%$) |
| **Dingling** | Regional Background | Other 11 | 34,213 | 44.15 | 60.08 | 0.3098 | 52.66 | 79.76 | -0.2165 | $-8.51\ \mu\text{g}/\text{m}^3$ ($-16.2\%$) |
| **Dongsi** | Urban Core | Other 11 | 34,242 | 48.03 | 67.32 | 0.3960 | 63.39 | 94.71 | -0.1955 | $-15.36\ \mu\text{g}/\text{m}^3$ ($-24.2\%$) |
| **Guanyuan** | Urban Core | Other 11 | 34,376 | 46.24 | 63.80 | 0.3791 | 59.15 | 88.16 | -0.1854 | $-12.91\ \mu\text{g}/\text{m}^3$ ($-21.8\%$) |
| **Gucheng** | Industrial Corridor | Other 11 | 34,346 | 46.97 | 67.24 | 0.3413 | 59.63 | 89.21 | -0.1596 | $-12.66\ \mu\text{g}/\text{m}^3$ ($-21.2\%$) |
| **Huairou** | Suburban | Other 11 | 34,039 | 43.45 | 59.30 | 0.3075 | 51.99 | 77.72 | -0.1892 | $-8.54\ \mu\text{g}/\text{m}^3$ ($-16.4\%$) |
| **Nongzhanguan** | Urban Core | Other 11 | 34,364 | 47.85 | 67.10 | 0.3951 | 63.03 | 94.72 | -0.2055 | $-15.18\ \mu\text{g}/\text{m}^3$ ($-24.1\%$) |
| **Shunyi** | Suburban | Other 11 | 34,079 | 47.45 | 66.56 | 0.3293 | 59.86 | 90.22 | -0.2321 | $-12.41\ \mu\text{g}/\text{m}^3$ ($-20.7\%$) |
| **Tiantan** | Urban Core | Other 11 | 34,315 | 46.27 | 64.26 | 0.3701 | 59.55 | 88.61 | -0.1977 | $-13.28\ \mu\text{g}/\text{m}^3$ ($-22.3\%$) |
| **Wanliu** | Urban Core | Other 11 | 34,610 | 47.10 | 65.34 | 0.3642 | 59.75 | 88.87 | -0.1761 | $-12.65\ \mu\text{g}/\text{m}^3$ ($-21.2\%$) |
| **Wanshouxigong**| Urban Core | Other 11 | 34,296 | 48.72 | 69.24 | 0.3522 | 62.66 | 93.87 | -0.1908 | $-13.94\ \mu\text{g}/\text{m}^3$ ($-22.2\%$) |
| **12-FOLD MEAN** | **Macro-Averaged** | — | **410,165** | **46.33 ± 1.72** | **64.48** | **0.3550** | **58.78** | **87.98** | **-0.1972** | **$-12.45\ \mu\text{g}/\text{m}^3$ ($-21.2\%$)** |

### Macro-Averaged vs. Pooled Definition:
- Reported aggregate metrics are **macro-averaged** (unweighted arithmetic mean of the 12 fold metrics).
- Because sample counts per station are closely balanced ($34,039$ to $34,610$), pooled metrics are within $0.05\ \mu\text{g}/\text{m}^3$ of macro-averaged values.

### Rejection of Universal Physical Transport Claims:
- Prior documentation claimed that consistent MAE across stations ($\sigma = 1.72\ \mu\text{g}/\text{m}^3$) "proves universal physical transport dynamics."
- **Correction:** Consistent cross-station performance demonstrates that the tree model relies primarily on invariant local physical relationships (wind velocity, atmospheric moisture, and autoregressive memory) rather than station identifiers. It does **not** prove that atmospheric transport is universally invariant. Unmeasured localized emission sources, street-canyon turbulence, and micro-climatic variations exist and cannot be proven invariant by tabular model performance alone.

---

## 7. Conformal Prediction Validity (Task 3 Audit)

### Theoretical Protocol Audit
- **Calibration Split Quarantine:** Calibration was performed on the 2016 validation partition ($N = 77,316$), keeping training data strictly separate from quantile estimation.
- **Finite-Sample Correction:** `_compute_quantile()` in `src/uncertainty/conformal.py` implements the standard inductive conformal formula:
  $$p = \frac{\lceil (n+1)(1-\alpha) \rceil}{n}$$
  and computes $\hat{q}$ via `np.quantile(scores, p, method="higher")`. This is mathematically correct.
- **The Exchangeability Violation:** The theoretical guarantee:
  $$\mathbb{P}(Y_{n+1} \in \hat{C}(X_{n+1})) \ge 1 - \alpha$$
  requires that calibration and test samples are exchangeable. In atmospheric time series, temporal correlation and seasonal shifts violate exchangeability.

### Empirical Stratified Conformal Audit Results (Nominal Coverage = 90.0%)

Below are the audited empirical coverage results evaluated directly on the test partition:

| Horizon ($h$) | Calibration Quantile ($\hat{q}_{0.90}$) | Overall Test Coverage | Mean Interval Width | Non-Heating (Oct) Coverage | Heating Season (Nov-Feb) Coverage | Extreme Episodes ($y \ge 150$) Coverage | Severe / Hazardous ($y \ge 250$) Coverage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h=1\text{h}$** | $18.46\ \mu\text{g}/\text{m}^3$ | **83.35%** | $34.93\ \mu\text{g}/\text{m}^3$ | 90.46% (Valid) | 81.61% (Under-coverage) | 69.89% | 48.85% |
| **$h=6\text{h}$** | $56.22\ \mu\text{g}/\text{m}^3$ | **79.59%** | $99.95\ \mu\text{g}/\text{m}^3$ | 88.79% (Near-valid) | 77.33% (Under-coverage) | 71.77% | 30.36% |
| **$h=24\text{h}$** | $85.51\ \mu\text{g}/\text{m}^3$ | **71.80%** | $157.84\ \mu\text{g}/\text{m}^3$ | 85.45% (Slight under) | 68.43% (Severe under) | 64.51% | **11.85% (Collapse)** |

### Stratified Breakdown by Station ($h=24\text{h}$, Nominal 90%)
- **Dingling (Background):** $79.48\%$ coverage (highest; cleaner air).
- **Huairou (Suburban):** $76.81\%$ coverage.
- **Changping (Suburban):** $74.96\%$ coverage.
- **Guanyuan (Urban Core):** $72.33\%$ coverage.
- **Aotizhongxin (Urban Core):** $72.31\%$ coverage.
- **Tiantan (Urban Core):** $71.97\%$ coverage.
- **Gucheng (Industrial):** $71.12\%$ coverage.
- **Wanliu (Urban Core):** $70.82\%$ coverage.
- **Dongsi (Urban Core):** $68.80\%$ coverage.
- **Shunyi (Suburban):** $68.78\%$ coverage.
- **Nongzhanguan (Urban Core):** $67.89\%$ coverage.
- **Wanshouxigong (Urban Core):** $66.29\%$ coverage (lowest; intense urban traffic).

### Scientific Diagnosis & Recommendations:
1. **Constant-Width vs. Heteroscedastic Error:** Split conformal prediction assigns a constant margin $\hat{q}$ across all feature inputs. In atmospheric science, error variance is heteroscedastic—proportional to particulate concentration. During clean periods ($y < 75\ \mu\text{g}/\text{m}^3$), coverage is over-conservative ($73.4\%\text{ to }98.5\%$). During severe episodes ($y > 250\ \mu\text{g}/\text{m}^3$), the error distribution expands, causing the constant margin to under-cover ($11.85\%$).
2. **Path to Resolution:** Production deployment must transition from standard split conformal prediction to **Conformalized Quantile Regression (CQR)** or **Adaptive Conformal Inference (ACI)** (Gibbs & Candès, 2021) to dynamically track non-stationary seasonal distribution shifts.

---

## 8. Explainability Limitations (Task 4 Audit)

### Verification of TreeSHAP Engine
- **Exact Implementation:** `ShapExplanationEngine` (`src/explainability/shap_engine.py`) calls XGBoost's native C++ `predict(pred_contribs=True)`, which implements Lundberg's exact polynomial-time TreeSHAP algorithm.
- **Meteorological Vector Convention:** In `WindVectorTransformer` (`src/preprocessing/transformers.py`):
  $$u = -WSPM \cdot \sin(\theta_{\text{rad}}), \quad v = -WSPM \cdot \cos(\theta_{\text{rad}})$$
  - For North wind ($\theta = 0^\circ$): air moves South $\implies u = 0, v = -WSPM < 0$.
  - For South wind ($\theta = 180^\circ$): air moves North $\implies u = 0, v = +WSPM > 0$.
  The negative attribution for $v < 0$ and positive attribution for $v > 0$ strictly matches meteorological transport physics.

### Explainability Limitations:
1. **Attribution is Not Causality:** SHAP values measure how much feature $j$ shifts the model prediction $\hat{f}(x)$ relative to the expected prediction $\mathbb{E}[\hat{f}(X)]$. They do not measure physical interventions in the real atmosphere.
2. **Collinearity and Credit Sharing:** Pollutants ($CO, NO_2, SO_2, PM_{10}$) and autoregressive lags share high collinearity. Tree ensembles greedily split on whichever collinear feature optimizes impurity at a given node; credit is partitioned among correlated predictors rather than reflecting isolated physical contributions.
3. **Language Standardization:** Claims that SHAP "proves" chemical mechanisms must be re-framed as: *"TreeSHAP demonstrates that the model utilizes surface wind speed and dew point depression in a manner consistent with known atmospheric dispersion and aerosol hygroscopic growth relationships."*

---

## 9. Robustness Verification (Task 5 Audit)

### Investigation of the 25% Gaussian Noise Result
- **Observation:** In Experiment 8, injecting $25\%$ Gaussian noise degraded test MAE by only $+0.26\%$ ($67.26 \to 67.43\ \mu\text{g}/\text{m}^3$).
- **Audit Findings:**
  1. **Post-Engineered Perturbation:** Noise was added directly to the 47 columns of `X_test`. It was not injected into raw sensor readings ($PM_{2.5}, TEMP, WSPM$) and propagated through the rolling/lag transformers.
  2. **Bounded Feature Distortions:** Unbounded Gaussian noise was added to cyclical harmonics (`hour_sin \in [-1, 1]`) and binary flags (`is_heating_season \in \{0, 1\}`), creating unphysical inputs that decision tree splits simply clamped to edge leaves.
  3. **Ensemble Averaging:** The XGBoost model comprises 160 shallow trees. Independent zero-mean symmetric noise across 47 features averages out across trees, leaving median predictions largely stable.
  4. **Single Seed:** The test evaluated a single seed (`seed=42`), lacking Monte Carlo variance bounds.

### Outage Stress Testing Findings
- **Telemetry Criticality:** The outage simulation verified that:
  - Loss of gas precursor telemetry ($SO_2, NO_2, CO, O_3$) caused only $\Delta\text{MAE} = +0.13\ \mu\text{g}/\text{m}^3$ degradation due to co-pollutant redundancy with particulate channels.
  - Loss of wind telemetry ($WSPM, u, v, wd$) caused a critical error surge of **$+2.05\ \mu\text{g}/\text{m}^3$** ($\text{MAE} = 69.30\ \mu\text{g}/\text{m}^3$). Wind vectors are non-redundant state variables.

---

## 10. API Integration Verification (Task 7 Audit)

### Architectural Audit
- **FastAPI Endpoints:** `api/main.py` exposes 10 verified routes:
  - `GET /health` (status, version, active horizons)
  - `GET /model-info` (metadata, registered forecasters)
  - `GET /features` (registered feature groups)
  - `GET /metrics` (experimental benchmark metrics)
  - `POST /predict` (single-sample prediction and interval)
  - `POST /predict/batch` (batch predictions)
- **Input Validation & Schemas:** Pydantic schemas in `api/schemas.py` enforce type constraints. In `api/services.py`, missing feature keys default cleanly to `0.0` without runtime exceptions.
- **Single vs. Batch Consistency:** Verified. Single and batch inference routes invoke identical model prediction logic.

### Serving Prototype Demarcation:
- **Heuristic Conformal Margins:** In `api/services.py`, prediction intervals use static empirical margins ($15, 35, 55\ \mu\text{g}/\text{m}^3$) scaled by a Gaussian multiplier rather than dynamically invoking the calibrated `ConformalPredictor` quantiles.
- **Prototype Status:** The API and Streamlit dashboard are **functional local prototypes**. They run on Uvicorn/Streamlit within a local Python environment. They do **not** feature distributed Redis caching, JWT authentication, rate limiting, or Prometheus monitoring.

---

## 11. Remaining Limitations

1. **Temporal Non-Stationarity in Conformal Intervals:** Split conformal prediction must be upgraded to Conformalized Quantile Regression (CQR) or Adaptive Conformal Inference (ACI) to handle seasonal distribution shifts.
2. **Tabular Sequence Representation in LSTM:** Recurrent models currently consume 1-step tabular vectors $(N, 1, 47)$ rather than multi-step tensor sequences $(N, T, F)$.
3. **Missing Telemetry Propagation in Stress Tests:** Noise injection should perturb raw telemetry at the sensor level and flow through the causal imputer and feature builder, rather than perturbing post-engineered matrices directly.
4. **Categorical Handling for Unseen Stations:** Unseen station identifiers in cross-station deployment should be represented using categorical entity embeddings or physical GIS coordinates (latitude, longitude, elevation) rather than arbitrary numeric sentinel values (`-1.0`).
5. **Partial Annual Cycle in Test Partition:** Future evaluations must incorporate a continuous multi-year test set covering all four meteorological seasons (Spring, Summer, Autumn, Winter).

---

## 12. Final Release Recommendation

### Verdict: **CONDITIONAL RELEASE (RESEARCH & PROTOTYPE GRADE)**

The AirCast Research framework is approved for release as an **Applied Machine Learning Research Project and Prototype Demonstration**. 

The system satisfies the highest standards of code cleanliness, test coverage, causal integrity, and reproducible scientific experimentation. 

### Mandatory Release Conditions:
1. **Documentation Alignment:** All research reports and the root `README.md` must clearly document the empirical conformal coverage limitations under seasonal distribution shift and distinguish observational associations from causal proof.
2. **Serving Demarcation:** The API and Streamlit applications must be explicitly labeled as research prototypes, and the heuristic interval calculation in `api/services.py` must be disclosed.
3. **No Retraining Mandate:** Existing trained model binaries and experimental metric JSON files are preserved as authentic, transparent records of the empirical research trajectory.
