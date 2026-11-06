# AirCast Research — Final Pre-Publish Scientific Consistency Audit

**Role:** Senior ML Systems Engineer & Release Quality Manager  
**Audit Date:** November 5, 2026  
**Commit Inspected:** `019adafd71b8bc80d88eb89ac98dbd955741f050`  
**Working Directory:** `beijing+multi+site+air+quality+data/`  
**Audit Purpose:** Comprehensive evidence-based reconciliation of conformal coverage, API route enumeration, latency profiling, and repository release integrity prior to public GitHub release.

---

## 1. Conformal Coverage Reconciliation

### 1.1 Comparison of Documentation and Artifacts

| Source File / Document | Scope / Method | +1h Coverage | +6h Coverage | +24h Coverage | Pooled Coverage | Severe Tier Coverage ($>250\ \mu\text{g}/\text{m}^3$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `artifacts/metrics/phase11_final_verified_results.json` | Normalized Adaptive Conformal | **89.07%** | **85.26%** | **79.81%** | **84.72%** | +1h: 86.34%<br/>+6h: 69.40%<br/>+24h: 28.05% |
| `artifacts/metrics/final_uncertainty_comparison.json` | Normalized Adaptive Conformal | **89.07%** | **85.26%** | **79.81%** | — | +1h: 87.82% ($>150$)<br/>+6h: 81.31% ($>150$)<br/>+24h: 56.24% ($>150$) |
| `artifacts/metrics/final_uncertainty_comparison.json` | Standard Homoscedastic Conformal | 83.37% | 79.71% | 70.21% | — | +1h: 48.75% ($>250$)<br/>+6h: 30.36% ($>250$)<br/>+24h: 11.85% ($>250$) |
| `artifacts/metrics/conformal_stratified_audit.json` | Standard Homoscedastic Conformal | 83.35% | 79.59% | 71.80% | — | +1h: 48.85% ($>250$)<br/>+6h: 30.36% ($>250$)<br/>+24h: 11.85% ($>250$) |
| `README.md` (Lines 261–271) | Normalized Adaptive Conformal | **89.07%** | **85.26%** | **79.81%** | **84.72%** | +1h: 86.34%<br/>+6h: 69.40%<br/>+24h: 28.05% |
| `research/release/final_github_release_report.md` | Release Verification Summary | Verified | Verified | Verified | Verified | Documented distribution shift |

### 1.2 Discrepancy Analysis: Root Cause of "91.31%, 91.80%, 91.06%"
- **Origin of the False Figures:**
  The figures `91.31%, 91.80%, 91.06%` do **not** exist in any metrics JSON file, script, or repository documentation. They were an erroneous hallucination generated in the previous conversational summary message (Turn 3104), where:
  1. The historical single-horizon Phase 7 metric ($91.4\%$ on mild autumn conditions) was conflated with multi-lead evaluations.
  2. The sub-stratum coverage of $91.81\%$ (which is the USG $[75, 115)$ tier coverage at $+6\text{h}$ in `final_uncertainty_comparison.json`) was mistakenly generalized across horizons.
- **Physical & Statistical Reality:**
  As proven by live execution of `scripts/audit_final_release.py` on the three saved calibrator artifacts (`artifacts/models/conformal_calibrator_h{1,6,24}.joblib`):
  - **+1h:** $89.07\%$ ($N = 42,554$, mean width: $44.88\ \mu\text{g}/\text{m}^3$, median width: $35.54\ \mu\text{g}/\text{m}^3$)
  - **+6h:** $85.26\%$ ($N = 42,494$, mean width: $131.29\ \mu\text{g}/\text{m}^3$, median width: $111.77\ \mu\text{g}/\text{m}^3$)
  - **+24h:** $79.81\%$ ($N = 42,278$, mean width: $181.15\ \mu\text{g}/\text{m}^3$, median width: $174.62\ \mu\text{g}/\text{m}^3$)
  - **Pooled Test Coverage:** **$84.72\%$** across all $127,326$ predictions.

### 1.3 Experimental Setup & Provenance Parameters
- **Calibration Method:** Normalized Split Adaptive Conformal Prediction:
  $$s_i = \frac{|y_i - \hat{y}_i|}{\sigma(x_i)}, \quad \text{where } \sigma(x_i) = \beta_0 + \beta_1 \hat{y}_i$$
  Scaling parameters: $\beta_0 = 5.0$, $\beta_1 \in [0.0892, 0.2140]$.
- **Calibration Partition:** Forward temporal split spanning April 15, 2016 – September 30, 2016 ($N \approx 42,500$ samples per horizon).
- **Test Partition:** Forward temporal holdout test partition spanning October 1, 2016 – February 28, 2017 ($N = 42,554$ for $+1\text{h}$; $N = 42,494$ for $+6\text{h}$; $N = 42,278$ for $+24\text{h}$).
- **Severe Non-Exchangeability Breakdown:**
  During severe winter heating episodes ($> 250\ \mu\text{g}/\text{m}^3$), residual variance surges due to nocturnal thermal inversions and stagnant boundary layers. Coverage contracts to:
  - $+1\text{h}$: $86.34\%$ ($N = 3,676$)
  - $+6\text{h}$: $69.40\%$ ($N = 3,676$)
  - $+24\text{h}$: $28.05\%$ ($N = 3,672$)
- **Primary Result Selection for README:**
  The README correctly displays the authentic empirical values: **89.07% (+1h), 85.26% (+6h), and 79.81% (+24h)**. No modification to `README.md` is required.

---

## 2. API Route Enumeration & Reconciliation

### 2.1 Complete List of Registered FastAPI Routes
Direct inspection of `api.main.app.routes` confirms **10 registered route objects**:

| Index | Path | HTTP Methods | Endpoint Function Name | Route Category |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `/openapi.json` | `GET`, `HEAD` | `openapi` | OpenAPI 3.1 JSON Specification |
| **2** | `/docs` | `GET`, `HEAD` | `swagger_ui_html` | Interactive Swagger UI Documentation |
| **3** | `/docs/oauth2-redirect` | `GET`, `HEAD` | `swagger_ui_redirect` | Swagger OAuth2 Authentication Redirect |
| **4** | `/redoc` | `GET`, `HEAD` | `redoc_html` | Interactive ReDoc Documentation |
| **5** | `/health` | `GET` | `health_check` | Service Diagnostics & Horizon Availability |
| **6** | `/model-info` | `GET` | `get_model_info` | Model Architecture Catalog & Metadata |
| **7** | `/features` | `GET` | `get_features` | Registered Feature Groups & Definitions |
| **8** | `/metrics` | `GET` | `get_benchmarking_metrics` | Empirical Experiment Benchmarks |
| **9** | `/predict` | `POST` | `predict` | Single-Record Point Forecast & Conformal Interval |
| **10** | `/predict/batch` | `POST` | `predict_batch` | Multi-Sample Batch Forecasting & Intervals |

### 2.2 Reconciliation of 9 vs 10 Routes
- **9-Route Count:** Obtained when counting **6 application business endpoints** (`/health`, `/model-info`, `/features`, `/metrics`, `/predict`, `/predict/batch`) plus **3 user-facing documentation routes** (`/openapi.json`, `/docs`, `/redoc`).
- **10-Route Count:** Obtained when programmatically querying `len(app.routes)`, which includes the internal Starlette/FastAPI OAuth callback handler `/docs/oauth2-redirect`.
- **Conclusion:** Both descriptions are scientifically accurate when scoped: **6 functional endpoints, 3 public documentation views, and 1 internal OAuth redirect handler (10 total routes in routing table)**.

---

## 3. Latency Profiling & Discrepancy Reconciliation

### 3.1 Profiling Across System Layers

```
Layer 1: Raw C++ / NumPy Kernel Inference (Preallocated buffer)
  Mean: 4.33 ms | P95: 7.43 ms (Single sample on CPU)
  Batch Throughput: 0.023 - 0.050 ms / sample (20,000 - 43,000 samples/sec)

Layer 2: In-Process Service Call (AirCastInferenceService.predict_single)
  +1h: Mean 28.05 ms | Median 35.98 ms | P95 44.13 ms | P99 57.86 ms
  +6h: Mean 35.04 ms | Median 38.00 ms | P95 43.36 ms | P99 66.70 ms
  +24h: Mean 24.72 ms | Median 13.84 ms | P95 43.32 ms | P99 48.43 ms
  (Attribution: ~86% single-row DataFrame DMatrix creation, ~14% feature parsing, <0.02 ms conformal)

Layer 3: REST API End-to-End HTTP Serving (api_performance.json)
  Mean: 71.86 ms | P50: 70.16 ms | P95: 99.66 ms | P99: 101.01 ms
  (Includes HTTP transport, JSON deserialization, Pydantic validation, serialization)
```

### 3.2 Audit of "2.91 ms, 1.87 ms, 1.45 ms"
In Turn 3104, the summary table reported "P95 Latency: 2.91 ms, 1.87 ms, 1.45 ms". Cross-referencing against the primary benchmark source (`artifacts/metrics/phase11_final_verified_results.json`) reveals the origin of this transcription error:
1. `2.91`: Lossless joblib compression ratio for XGBoost (`xgboost_level3: 2.91x`).
2. `1.87`: RAM downcasting factor reduction (`factor_reduction: 1.87x`) or structural RF pruning disk footprint (`1.87 MB`).
3. `1.45`: Floating-point fragment from float32 RAM footprint ($121.45\text{ MB}$) or direct LSTM mean inference ($1.49\text{ ms}$).

**Corrected Latency Ground Truth:**
- **Raw C++ Model Execution (P95):** **$7.43\text{ ms}$** (Mean: $4.33\text{ ms}$).
- **In-Process Service Steady-State (P95):** **$44.13\text{ ms}$** (+1h), **$43.36\text{ ms}$** (+6h), **$43.32\text{ ms}$** (+24h).
- **REST End-to-End HTTP Service (P95):** **$99.66\text{ ms}$** (Mean: $71.86\text{ ms}$).
- **Budget Compliance:** The $\le 15.0\text{ ms}$ SLA is **MET** for raw C++ NumPy inference ($4.33\text{ ms}$), but **UNMET** for single-record REST API endpoints ($43\text{--}44\text{ ms}$ service, $99.7\text{ ms}$ HTTP). `README.md` Section 14 already explicitly discloses this SLA status.

---

## 4. Final System Verification Results

### 4.1 Pytest Automated Regression Suite
- **Command:** `python -m pytest tests/ -q`
- **Results:** `52 passed, 6 warnings in 39.62s`
- **Pass Rate:** **100.0% (52/52)**
  - `tests/test_api.py`: 5 passed
  - `tests/test_conformal.py`: 5 passed
  - `tests/test_efficiency.py`: 6 passed
  - `tests/test_evaluation.py`: 6 passed
  - `tests/test_explainability.py`: 4 passed
  - `tests/test_feature_engineering.py`: 8 passed
  - `tests/test_models.py`: 8 passed
  - `tests/test_preprocessing.py`: 5 passed
  - `tests/test_robustness.py`: 5 passed

### 4.2 Documentation Link Integrity
- **Total Markdown Links in `README.md`:** 43 (34 local file references, 9 external web URLs).
- **Broken Local Links:** **0 (100% valid)**.

### 4.3 Git State Audit
- **Branch:** `main`
- **Commit Hash:** `019adafd71b8bc80d88eb89ac98dbd955741f050`
- **Working Tree:** Clean (`nothing to commit, working tree clean`)
- **Tracked Files:** 160 files ($45.93\text{ MB}$)
- **Largest Tracked File:** `data/processed/forecasting_features_cache.parquet` ($30.52\text{ MB} \ll 50\text{ MB}$)
- **Git LFS Requirement:** None.

---

## 5. Pre-Publish Sign-Off Summary

1. **Conformal Coverage:** Verified at **89.07% (+1h), 85.26% (+6h), 79.81% (+24h)**, with **84.72% pooled** test coverage. README accurately reflects these numbers.
2. **API Routes:** Verified at **10 registered routes** (6 application endpoints, 3 documentation routes, 1 internal redirect).
3. **Latency Benchmarks:** Verified and cleanly delineated into raw C++ kernel ($4.33\text{ ms}$ mean / $7.43\text{ ms}$ P95), in-process service ($43.3\text{--}44.1\text{ ms}$ P95), and REST end-to-end ($99.66\text{ ms}$ P95).
4. **Publish Readiness:** 100% verified. No files pushed, no models modified. Ready for publication whenever the user chooses.
