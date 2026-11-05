# AirCast Research — Stage 8: Serving, API & Dashboard Performance Report

**Author:** Principal Applied Scientist & ML Performance Engineer  
**Project:** AirCast Research: A Research-Driven Framework for Explainable, Uncertainty-Aware and Generalizable Multi-Horizon Air Quality Forecasting  
**Date:** March 2026  
**Artifacts Generated:**
- `artifacts/metrics/api_performance.json`
- `artifacts/metrics/dashboard_performance.json`

---

## 1. Executive Summary

Stage 8 of Phase 11 profiled and stress-tested the serving infrastructure, focusing on the RESTful FastAPI application (`api/main.py`), Pydantic v2 data contract validation, concurrency scaling under multi-threaded client load, and Streamlit interactive dashboard caching efficiency.

### Key Empirical Findings:
1. **REST Endpoint Latency:**
   - **`/health`:** Mean latency **$10.16\text{ ms}$** (P95: $29.72\text{ ms}$).
   - **`/model-info`:** Mean latency **$13.73\text{ ms}$** (P95: $28.39\text{ ms}$) across dynamic catalog inspection.
   - **`/predict` (Single Forecast + Conformal Interval):** Mean latency **$71.86\text{ ms}$** (P95: $99.66\text{ ms}$).
2. **Batch Endpoint Vectorization Scaling (`/predict/batch`):**
   - At $B = 100$ samples, batch throughput reaches **$1,079\text{ samples/sec}$** ($0.926\text{ ms}$ per sample).
   - At $B = 500$ samples, batch throughput reaches **$3,911\text{ samples/sec}$** ($0.256\text{ ms}$ per sample). Total batch latency is only **$127.84\text{ ms}$**, proving strong sub-linear scaling ($O(1)$ amortized overhead).
3. **Concurrency & Load Profiling:**
   - Single-worker Python FastAPI processes saturate at ~15 requests/sec under CPU-bound tree inference.
4. **Dashboard Caching Performance:**
   - In-memory Streamlit caching (`@st.cache_data`) reduces UI rerender latency by **$16.7\times$ to $1,975\times$**, cutting artifact load times from $2.77\text{ ms}$ down to $0.0014\text{ ms}$.

---

## 2. API Endpoint Profiling

Benchmark performed across $N = 100$ requests per endpoint using HTTP client test harness.

| Endpoint | HTTP Method | Mean Latency (ms) | Median (P50) (ms) | P95 Latency (ms) | P99 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`/health`** | GET | 10.158 | 8.871 | 29.724 | 33.205 |
| **`/model-info`** | GET | 13.725 | 12.189 | 28.394 | 31.810 |
| **`/predict`** | POST | 71.858 | 70.923 | 99.657 | 114.230 |

---

## 3. Batch Endpoint Scaling (`/predict/batch`)

| Batch Size ($B$) | Total Latency (ms) | Latency per Sample (ms) | Throughput (Samples / Sec) |
| :---: | :---: | :---: | :---: |
| **10** | 78.29 | 7.829 | 128 |
| **50** | 83.03 | 1.661 | 602 |
| **100** | 92.64 | 0.926 | 1,079 |
| **250** | 107.78 | 0.431 | 2,320 |
| **500** | 127.84 | **0.256** | **3,911** |

Notice that moving from a batch size of 10 to 500 only increases total request duration from $78.29\text{ ms}$ to $127.84\text{ ms}$ ($+63.3\%$ time for $50\times$ more data), demonstrating the massive efficiency of vectorized batch evaluation in production.

---

## 4. Concurrency Profiling

Client concurrency tested using thread pool simulation across 50 requests per level:

| Concurrency Level | Wall-Clock Time (s) | System Throughput (Req / Sec) | Mean Request Latency (ms) | P95 Latency (ms) |
| :---: | :---: | :---: | :---: | :---: |
| **1 client** | 3.65 | 13.7 | 72.87 | 99.87 |
| **5 clients** | 3.79 | 13.2 | 368.35 | 488.04 |
| **10 clients** | 3.31 | 15.1 | 624.28 | 939.38 |
| **25 clients** | 3.55 | 14.1 | 1471.18 | 2247.59 |

**Production Architecture Recommendation:** Because Python GIL bounds single-process throughput to ~15 req/s under synchronous inference, production deployment should place Gunicorn with 4 Uvicorn worker processes behind an NGINX reverse proxy to achieve $> 60\text{ req/sec}$ throughput.

---

## 5. Dashboard Optimization

Evaluating Streamlit memory artifact retrieval times:

| Artifact | Size | Cold Disk Read Time | Cached Memory Access | Speedup Factor |
| :--- | :---: | :---: | :---: | :---: |
| `efficiency_baseline.json` | 37.4 KB | 2.766 ms | 0.0014 ms | **$1,975\times$** |
| `feature_efficiency_comparison.json` | 25.0 KB | 1.942 ms | 0.1072 ms | **$18.1\times$** |
| `model_complexity_optimization.json` | 11.1 KB | 1.521 ms | 0.0913 ms | **$16.7\times$** |

Applying `@st.cache_data` completely eliminates disk I/O bottlenecks during user interaction with the dashboard.
