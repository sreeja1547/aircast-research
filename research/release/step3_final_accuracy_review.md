# AirCast Research — Step 3.1: Final README Scientific Accuracy Review

**Role:** Senior Applied Scientist, Machine Learning Systems Architect & Statistical Auditor  
**Date:** November 2026  
**Status:** Complete & Formally Verified  
**Audited Documents:**  
- [`README.md`](../../README.md)  
- [`LICENSE`](../../LICENSE)  
- [`artifacts/metrics/phase11_final_verified_results.json`](../../artifacts/metrics/phase11_final_verified_results.json)  
- [`artifacts/metrics/final_sensor_robustness.json`](../../artifacts/metrics/final_sensor_robustness.json)  
- [`artifacts/metrics/experiment8_robustness.json`](../../artifacts/metrics/experiment8_robustness.json)  
- [`api/main.py`](../../api/main.py)  

---

## 1. SHAP & Atmospheric Interpretation Audit

### 1.1 Scope & Motivation
TreeSHAP computes exact path-dependent Shapley feature attributions within an empirical gradient boosted tree ensemble. Because Shapley values reflect feature importance within the statistical model representation—rather than counterfactual physical causation—all descriptions in project documentation must adhere to association-based scientific language. Physical interpretations are framed as empirical consistency with established meteorological hypotheses, avoiding causal overclaiming.

### 1.2 Language Modifications & Scientific Reconciliations

| Domain Topic | Unreconciled / Causal Phrasing | Reconciled Scientific / Association-Based Phrasing | Scientific Justification |
| :--- | :--- | :--- | :--- |
| **Model Verification** | *"proves models capture genuine physical transport mechanisms"* | *"evaluates model alignment with domain atmospheric transport hypotheses"* | SHAP measures empirical model attribution conditioned on data distribution, not formal causal proof of atmospheric transport. |
| **Northerly Ventilation** | *"cleansing the urban basin"* | *"associated with mountain ventilation pathways... statistically consistent with clean air dispersion hypotheses"* | Identifies the physical mechanism as an established atmospheric hypothesis consistent with negative SHAP attributions ($-20$ to $-45\ \mu\text{g}/\text{m}^3$). |
| **Southerly Advection** | *"advects industrial emissions from Hebei Province"* | *"align with documented regional transport corridors from the industrial south"* | Frame as empirical alignment with known geographic emission sources rather than instrumentally observed particle trajectory tracking. |
| **Hygroscopic Growth** | *"due to secondary sulfate/nitrate aqueous-phase conversion"* | *"consistent with the atmospheric hypothesis of accelerated secondary sulfate/nitrate aqueous-phase formation and hygroscopic aerosol swelling under high ambient moisture"* | Observational telemetry lacks chemical speciation instruments; model attribution shows non-linear sensitivity at $T - T_{\text{dew}} < 2^\circ\text{C}$ ($RH > 85\%$). |
| **Heating-Season Baseline** | *"shifts baseline forecasts upward by $+15\ \mu\text{g}/\text{m}^3$"* | *"contributes positive baseline SHAP offsets (up to $+15\ \mu\text{g}/\text{m}^3$)... reflecting empirical alignment with seasonal coal-combustion emission increases"* | Distinguish model additive attribution offset from physical source-apportionment chemical measurements. |

---

## 2. Robustness Experiment Reconciliation

### 2.1 Artifact Traceability & Experiment Lineage
Two distinct robustness experiments exist in the AirCast repository:

1. **Phase 8 Preliminary Baseline Robustness (`artifacts/metrics/experiment8_robustness.json`):**
   - *Model:* Baseline unpruned XGBoost ($180$ trees, max depth $7$) at $h=+24\text{h}$.
   - *Test Set:* $N = 42,278$ observations.
   - *Methodology:* Noise was injected directly into precomputed tabular feature columns without causal re-derivation of autoregressive lags or rolling windows. Evaluated across a single seed.
   - *Results:* $25\%$ noise yielded $+0.26\%$ degradation; wind outage yielded $\Delta\text{MAE} = +2.05\ \mu\text{g}/\text{m}^3$; precursor outage yielded $\Delta\text{MAE} = +0.13\ \mu\text{g}/\text{m}^3$.

2. **Phase 11 Reconciled Controlled Sensor-Perturbation Robustness (`artifacts/metrics/final_sensor_robustness.json` & `research/final_correction_report.md` Section 4):**
   - *Model:* Recommended production forecaster `xgboost_fast_h24.joblib` ($50$ trees, max depth $5$) at $h=+24\text{h}$.
   - *Test Set:* $N = 42,278$ observations.
   - *Methodology:* Zero-mean Gaussian noise was injected directly into **raw telemetry instrument channels** ($\{PM_{2.5}, PM_{10}, SO_2, NO_2, CO, O_3, TEMP, PRES, DEWP, WSPM\}$). All $47$ causal features (wind vectors $u, v$, dew depression, 6 autoregressive lags, and 12 rolling statistics) were **dynamically re-derived** from the perturbed series. Evaluated across **3 independent random seeds** ($\{42, 123, 999\}$).
   - *Results:* Clean baseline MAE is $66.5956\ \mu\text{g}/\text{m}^3$.

### 2.2 Reconciled Multi-Seed Measurement Tables

#### Multi-Channel Gaussian Sensor Noise ($h=+24\text{h}$, 3 Seeds)
*Source: `artifacts/metrics/final_sensor_robustness.json`*

| Noise Ratio ($\rho$) | Mean Test MAE ($\mu\text{g}/\text{m}^3$) | MAE Std ($\pm$) | Mean Test RMSE ($\mu\text{g}/\text{m}^3$) | Mean $R^2$ | Mean Degradation (%) | Supported Claim in README |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0% (Clean Baseline)** | **$66.5956$** | — | $92.6649$ | $0.2106$ | $0.00\%$ | Reference baseline |
| **5% Raw Noise** | $66.6715$ | $\pm 0.0162$ | $92.7376$ | $0.2093$ | $+0.11\%$ | Negligible error drift |
| **15% Raw Noise** | $66.9763$ | $\pm 0.0348$ | $93.0032$ | $0.2048$ | $+0.57\%$ | Stable under sensor drift |
| **25% Raw Noise** | $67.4896$ | $\pm 0.0728$ | $93.5322$ | $0.1957$ | **$+1.34\%$** | $\Delta\text{MAE} = +0.89\ \mu\text{g}/\text{m}^3$ |

#### Hardware Outage & Telemetry Disconnection Hierarchy ($h=+24\text{h}$)
*Source: `artifacts/metrics/final_sensor_robustness.json`*

| Outage Scenario | Dropped Signals | Test MAE ($\mu\text{g}/\text{m}^3$) | Test RMSE ($\mu\text{g}/\text{m}^3$) | Test $R^2$ | $\Delta\text{MAE}$ | Degradation (%) | Severity Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clean Baseline** | 0 | $66.60$ | $92.66$ | $0.2106$ | $0.00$ | $0.00\%$ | Reference |
| **CO Sensor Failure** | 1 | $66.58$ | $92.43$ | $0.2145$ | $-0.01$ | $-0.02\%$ | Negligible / Neutral |
| **Complete Precursor Gas Failure** | 4 | $66.90$ | $92.14$ | $0.2194$ | $+0.30$ | $+0.46\%$ | Low Sensitivity |
| **Wind Anemometer Suite Failure** | 5 | $68.66$ | $93.55$ | $0.1954$ | $+2.06$ | $+3.09\%$ | Significant Sensitivity |
| **Particulate Sensor Disconnection**| 24 | **$72.28$** | **$99.18$** | **$0.0956$** | **$+5.68$** | **$+8.53\%$** | Critical Sensitivity |

### 2.3 Reconciliation Decision
The README previously cited $+0.26\%$ from Phase 8 alongside values from Phase 11. In the revised `README.md`, Section 13 strictly presents the **controlled sensor-perturbation multi-seed evaluation** ($+1.34\%$ degradation under $25\%$ raw noise across 3 seeds), while explicitly adding the particulate disconnection baseline ($72.28\ \mu\text{g}/\text{m}^3$, which still beats naive persistence at $85.78\ \mu\text{g}/\text{m}^3$) and contextualizing the Phase 8 preliminary measurement.

---

## 3. FastAPI Route Count & Classification Verification

### 3.1 Codebase Inspection (`api/main.py`)
Direct AST inspection of `api/main.py` confirms that exactly 6 route decorators are defined on the FastAPI application instance `app`:

```python
@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
@app.get("/model-info", response_model=ModelCatalogResponse, tags=["Metadata"])
@app.get("/features", response_model=FeatureRegistryResponse, tags=["Features"])
@app.get("/metrics", tags=["Diagnostics"])
@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
```

### 3.2 Delineation of Route Classifications

1. **Functional Application Endpoints (6 Total):**
   - `GET /health`: Diagnostic probe returning service health, process uptime, and memory consumption.
   - `GET /model-info`: Active model catalog detailing artifact filenames, parameter counts, and disk footprints.
   - `GET /features`: Schema registry defining the 47 expected features, data types, and required ranges.
   - `GET /metrics`: Multi-horizon benchmark performance summary.
   - `POST /predict`: Single-record forecasting endpoint returning point prediction and adaptive conformal intervals.
   - `POST /predict/batch`: Vectorized multi-record batch forecasting endpoint.

2. **Framework Documentation & OpenAPI Schema Routes (3 Total):**
   - `GET /docs`: Swagger UI interactive documentation.
   - `GET /redoc`: ReDoc interactive technical documentation.
   - `GET /openapi.json`: Standard OpenAPI 3.1.0 JSON schema.

**Total Registered Routes:** **9 routes** (6 functional application endpoints + 3 documentation/schema routes).  
The revised `README.md` explicitly separates these two categories into distinct tables.

---

## 4. MIT License & Third-Party Ownership Review

### 4.1 License File Inspection (`LICENSE`)
The root `LICENSE` file contains the complete, standard text of the MIT License:
- Correctly formatted with standard disclaimer of warranty and limitation of liability.
- Attributed to: `Copyright (c) 2026 AirCast Applied Sciences Research Team`.

### 4.2 Scope Boundaries & Third-Party Data Disclosures
To ensure intellectual property integrity, the license terms are scoped as follows:

1. **Repository Source Code & Documentation:**
   - All original Python source code in `src/`, `api/`, `dashboard/`, `scripts/`, `tests/`, and technical reports in `research/` are licensed under the MIT License.

2. **Atmospheric Dataset Attribution (PRSA 2017):**
   - The telemetry dataset (`PRSA_Data_20130301-20170228/`) was created and curated by Song Chen et al. and hosted by the UCI Machine Learning Repository:
     > Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). *Cautionary tales on air-quality improvement in Beijing.* Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2205), 20170457.
   - **Zero Claim of Ownership:** The authors and contributors of AirCast make no claim of ownership or copyright over the raw PRSA 2017 dataset. Its inclusion is strictly for scientific reproducibility under academic open data terms.

3. **Third-Party Python Dependencies:**
   - Software packages used (NumPy, Pandas, Scipy, PyArrow, Joblib, Scikit-Learn, XGBoost, PyTorch, FastAPI, Uvicorn, Pydantic, Streamlit, Matplotlib, Seaborn, Pytest) remain subject to their respective open-source licenses (BSD, Apache 2.0, MIT).

---

## 5. Comprehensive Validation & Verification Results

### 5.1 Documentation Link Integrity Check
Every relative link in `README.md` was validated using automated filesystem resolution:
- **Total local links tested:** 31
- **Broken / dangling links:** **0 (100% valid)**
- All links to technical markdown reports in `research/`, `research/efficiency/`, `research/release/`, tests, and the `LICENSE` file resolve to real files.

### 5.2 Metrics Consistency Audit
All reported predictive metrics in `README.md` match primary JSON artifacts (`artifacts/metrics/phase11_final_verified_results.json`):

| Horizon | Recommended Model | Test Samples ($N$) | Verified Test MAE | Verified Test RMSE | Verified Test $R^2$ | Primary JSON Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$+1\text{h}$** | `xgboost_h1.joblib` | 42,554 | 11.5723 | 23.3344 | 0.9499 | **Bit-Identical** |
| **$+6\text{h}$** | `xgboost_pruned_h6.joblib` | 42,494 | 37.0350 | 62.6263 | 0.6389 | **Bit-Identical** |
| **$+24\text{h}$** | `xgboost_fast_h24.joblib` | 42,278 | 66.0738 | 92.5722 | 0.2121 | **Bit-Identical** |

### 5.3 Automated Pytest Regression Verification
The complete test suite was executed against the local environment (Python 3.13.5, Windows 11 x64):

```text
pytest tests/ -q
.................................................... [100%]
52 passed, 6 warnings in 42.45s
```

**Results:** **52 / 52 Passed (100.0% Pass Rate).**  
Zero test regressions detected across feature engineering, multi-horizon models, conformal prediction, TreeSHAP explainability, 12-fold LOSO, robustness stress testing, and API serving.

---

## 6. Audit Conclusion & Release Gate Sign-Off

The final scientific accuracy review confirms:
1. All atmospheric and SHAP interpretations use rigorous, association-based scientific language.
2. The robustness section accurately presents multi-seed controlled sensor-perturbation findings with full parameter documentation.
3. The API route catalog correctly delineates 6 functional endpoints from 3 framework documentation routes.
4. The MIT License and third-party data attribution properly protect intellectual property and credit original data authors.
5. All 31 markdown links, tables, and metrics are verified bit-for-bit against primary project artifacts.

**Release Status:** **Step 3.1 Review Complete. Halted before Git initialization.**
