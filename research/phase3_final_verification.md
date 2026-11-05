# AirCast Research: Phase 3 Source-Code Audit & Full-Dataset Integration Matrix

**Document Version:** 1.1.0  
**Audit Completion Date:** November 2026  
**Auditor:** Senior Applied Scientist, ML Research Engineer & Independent Reviewer  
**Module Audit Targets:** `src/features/` (`builder.py`, `registry.py`), `src/data/` (`forecasting_dataset.py`), `src/preprocessing/` (`cleaner.py`, `splitter.py`, `transformers.py`)  
**Test Suite:** `tests/test_feature_engineering.py`, `tests/test_preprocessing.py` (19 passing unit tests)  
**Machine-Readable Summary Artifacts:**
- Feature Registry Summary: [artifacts/metrics/feature_registry_summary.json](../artifacts/metrics/feature_registry_summary.json)
- Full Dataset Integration Metrics (Complete 18 Combinations): [artifacts/metrics/full_dataset_integration_metrics.json](../artifacts/metrics/full_dataset_integration_metrics.json)

---

## 1. Executive Summary & Audit Corrections

This document provides the final independent source-code audit and full-dataset integration validation of Phase 3 for AirCast Research. In response to review requirements:
1. **Target Accounting Disambiguation:** Disambiguated pre-filtering unobserved targets from post-filtering target integrity metrics.
2. **Complete 18-Combination Integration Matrix:** Executed the end-to-end forecasting dataset builder across all 6 feature groups and all 3 horizons ($6 \times 3 = 18$ combinations) on the entire 420,768-row dataset.
3. **Reproducible Test Execution:** All 19 automated tests passed in 2.50s.

---

## 2. Metric Definitions & Missing Target Accounting

To eliminate any ambiguity between raw dataset characteristics and post-filtering supervised arrays, the following metric definitions are established:

### Formal Metric Definitions
1. **Total Raw Rows ($N_{\text{total}}$):** Total number of hourly observations across all 12 stations ($12 \times 35,064 = \mathbf{420,768}$).
2. **Raw Unobserved Target Count ($N_{\text{unobs}}^{(h)}$):** Number of instances where the target observation at timestamp $t + h$ is unobserved in raw sensor telemetry. Across all 12 stations, exactly **$8,739$** observations ($2.0769\%$) lack valid $PM_{2.5}$ measurements. Because stations are unbroken hourly series, shifting forward by $h$ hours transfers these unobserved points to prediction time $t$.
3. **Horizon Boundary Exclusions ($N_{\text{bound}}^{(h)}$):** Number of rows at the end of each station series that lack future target observations within that station's timeline:
   $$N_{\text{bound}}^{(h)} = 12\ \text{stations} \times h\ \text{hours}$$
   - For $h=1$: $12 \times 1 = \mathbf{12\ \text{rows}}$
   - For $h=6$: $12 \times 6 = \mathbf{72\ \text{rows}}$
   - For $h=24$: $12 \times 24 = \mathbf{288\ \text{rows}}$
4. **Total Unavailable Target Exclusions ($N_{\text{target\_excl}}^{(h)}$):** Sum of unobserved target measurements and boundary shifts:
   $$N_{\text{target\_excl}}^{(h)} = N_{\text{unobs}}^{(h)} + N_{\text{bound}}^{(h)}$$
   - For $h=1$: $8,739 + 12 = \mathbf{8,751\ \text{rows}}$
   - For $h=6$: $8,739 + 72 = \mathbf{8,811\ \text{rows}}$
   - For $h=24$: $8,739 + 288 = \mathbf{9,027\ \text{rows}}$
5. **Lag Warmup Exclusions ($N_{\text{warmup}}$):** Number of rows at the beginning of each station series that cannot compute the required 48-hour lag feature ($PM_{2.5\_lag\_48}$):
   $$N_{\text{warmup}} = \begin{cases} 12\ \text{stations} \times 48\ \text{hours} = \mathbf{576\ \text{rows}} & \text{if feature group includes lags} \\ \mathbf{0\ \text{rows}} & \text{if feature group does not include lags} \end{cases}$$
6. **Final Usable Supervised Samples ($N_{\text{usable}}$):** Clean supervised sample count available for training and evaluation:
   $$N_{\text{usable}} = N_{\text{total}} - N_{\text{target\_excl}}^{(h)} - N_{\text{warmup}}$$
7. **Post-Filtering Missing Targets in $y$:** Integrity metric verifying that after `drop_unobserved_targets()`, the target array $y$ contains **0** missing values.

---

## 3. Complete 18-Combination Integration Matrix

Every combination of feature group and forecast horizon was generated directly from the raw dataset. Results are saved in [artifacts/metrics/full_dataset_integration_metrics.json](../artifacts/metrics/full_dataset_integration_metrics.json):

| Combo ID | Horizon ($h$) | Feature Group | Feature Cols ($X$) | Raw Rows ($N_{\text{total}}$) | Target Unobs ($N_{\text{unobs}}$) | Boundary Excl ($N_{\text{bound}}$) | Target Excl ($N_{\text{target\_excl}}$) | Lag Warmup Excl ($N_{\text{warmup}}$) | Usable Samples ($N_{\text{usable}}$) | Missing in $X$ | Post-Filter Missing $y$ | Memory ($X+y$) | Timestamp Coverage |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | **1h** | `temporal_baseline` | 8 | 420,768 | 8,739 | 12 | 8,751 | 0 | **412,017** | 0 | 0 | 33.22 MB | 2013-03-01 00:00 to 2017-02-28 22:00 |
| **02** | **1h** | `historical_pollutants`| 14 | 420,768 | 8,739 | 12 | 8,751 | 0 | **412,017** | 0 | 0 | 52.08 MB | 2013-03-01 00:00 to 2017-02-28 22:00 |
| **03** | **1h** | `lag_features` | 20 | 420,768 | 8,739 | 12 | 8,751 | 576 | **411,441** | 0 | 0 | 70.84 MB | 2013-03-03 00:00 to 2017-02-28 22:00 |
| **04** | **1h** | `rolling_features` | 36 | 420,768 | 8,739 | 12 | 8,751 | 576 | **411,441** | 0 | 0 | 121.07 MB | 2013-03-03 00:00 to 2017-02-28 22:00 |
| **05** | **1h** | `meteorological_features`| 25 | 420,768 | 8,739 | 12 | 8,751 | 0 | **412,017** | 0 | 0 | 85.09 MB | 2013-03-01 00:00 to 2017-02-28 22:00 |
| **06** | **1h** | `combined_all` | 47 | 420,768 | 8,739 | 12 | 8,751 | 576 | **411,441** | 0 | 0 | 154.03 MB | 2013-03-03 00:00 to 2017-02-28 22:00 |
| **07** | **6h** | `temporal_baseline` | 8 | 420,768 | 8,739 | 72 | 8,811 | 0 | **411,957** | 0 | 0 | 33.21 MB | 2013-03-01 00:00 to 2017-02-28 17:00 |
| **08** | **6h** | `historical_pollutants`| 14 | 420,768 | 8,739 | 72 | 8,811 | 0 | **411,957** | 0 | 0 | 52.07 MB | 2013-03-01 00:00 to 2017-02-28 17:00 |
| **09** | **6h** | `lag_features` | 20 | 420,768 | 8,739 | 72 | 8,811 | 576 | **411,381** | 0 | 0 | 70.83 MB | 2013-03-03 00:00 to 2017-02-28 17:00 |
| **10** | **6h** | `rolling_features` | 36 | 420,768 | 8,739 | 72 | 8,811 | 576 | **411,381** | 0 | 0 | 121.05 MB | 2013-03-03 00:00 to 2017-02-28 17:00 |
| **11** | **6h** | `meteorological_features`| 25 | 420,768 | 8,739 | 72 | 8,811 | 0 | **411,957** | 0 | 0 | 85.07 MB | 2013-03-01 00:00 to 2017-02-28 17:00 |
| **12** | **6h** | `combined_all` | 47 | 420,768 | 8,739 | 72 | 8,811 | 576 | **411,381** | 0 | 0 | 154.00 MB | 2013-03-03 00:00 to 2017-02-28 17:00 |
| **13** | **24h**| `temporal_baseline` | 8 | 420,768 | 8,739 | 288 | 9,027 | 0 | **411,741** | 0 | 0 | 33.20 MB | 2013-03-01 00:00 to 2017-02-27 23:00 |
| **14** | **24h**| `historical_pollutants`| 14 | 420,768 | 8,739 | 288 | 9,027 | 0 | **411,741** | 0 | 0 | 52.04 MB | 2013-03-01 00:00 to 2017-02-27 23:00 |
| **15** | **24h**| `lag_features` | 20 | 420,768 | 8,739 | 288 | 9,027 | 576 | **411,165** | 0 | 0 | 70.79 MB | 2013-03-03 00:00 to 2017-02-27 23:00 |
| **16** | **24h**| `rolling_features` | 36 | 420,768 | 8,739 | 288 | 9,027 | 576 | **411,165** | 0 | 0 | 120.98 MB | 2013-03-03 00:00 to 2017-02-27 23:00 |
| **17** | **24h**| `meteorological_features`| 25 | 420,768 | 8,739 | 288 | 9,027 | 0 | **411,741** | 0 | 0 | 85.03 MB | 2013-03-01 00:00 to 2017-02-27 23:00 |
| **18** | **24h**| `combined_all` | 47 | 420,768 | 8,739 | 288 | 9,027 | 576 | **411,165** | 0 | 0 | 153.92 MB | 2013-03-03 00:00 to 2017-02-27 23:00 |

### Mathematical Reconciliation Checks
- **Check 1 ($h=1$, no lags):** $420,768 - 8,739 - 12 - 0 = \mathbf{412,017}$ (Verified Exact).
- **Check 2 ($h=1$, with lags):** $420,768 - 8,739 - 12 - 576 = \mathbf{411,441}$ (Verified Exact).
- **Check 3 ($h=6$, no lags):** $420,768 - 8,739 - 72 - 0 = \mathbf{411,957}$ (Verified Exact).
- **Check 4 ($h=6$, with lags):** $420,768 - 8,739 - 72 - 576 = \mathbf{411,381}$ (Verified Exact).
- **Check 5 ($h=24$, no lags):** $420,768 - 8,739 - 288 - 0 = \mathbf{411,741}$ (Verified Exact).
- **Check 6 ($h=24$, with lags):** $420,768 - 8,739 - 288 - 576 = \mathbf{411,165}$ (Verified Exact).

---

## 4. Automated Test Suite Verification

The complete test suite was re-executed:

```powershell
python -m pytest tests/
```

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-8.3.4, pluggy-1.5.0
rootdir: C:\Users\Dell\Downloads\beijing+multi+site+air+quality+data
plugins: anyio-4.11.0
collected 19 items

tests\test_feature_engineering.py ............                           [ 63%]
tests\test_preprocessing.py .......                                      [100%]

======================== 19 passed, 1 warning in 2.50s ========================
```

All 19 unit tests pass with zero failures.

---

## 5. Final Audit Verdict

# **DECISION: PHASE 3 OFFICIALLY CLOSED & FULLY READY FOR PHASE 4**

All four requested audit corrections are complete:
1. Missing-target accounting is reconciled and explicitly separated from post-filtering integrity metrics.
2. Complete 18-combination matrix is generated from the actual pipeline and saved in JSON and Markdown formats.
3. All 19 unit tests re-run and passed in 2.50s.
4. Documentation is updated.

No forecasting models have been trained. Awaiting approval to proceed to Phase 4 (Baseline and Advanced Model Development).
