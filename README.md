# AirCast Research

### A Research-Driven Framework for Explainable, Uncertainty-Aware, and Generalizable Multi-Horizon Air Quality Forecasting

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-orange)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-76%20Passed-success)](https://github.com/sreeja1547/aircast-research)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AirCast Research** is an applied machine learning framework for forecasting atmospheric fine particulate matter (PM2.5) across multiple future time horizons. It combines causal feature engineering, XGBoost forecasting, model explainability, normalized split conformal prediction, spatial generalization experiments, and CPU inference optimization.

The project uses the Beijing Multi-Site Air-Quality Dataset and provides an interactive Streamlit research dashboard and a FastAPI inference service.

## Live Demo

**[Open AirCast Research Dashboard](https://aircast-research-1.onrender.com/)**

The interactive dashboard provides access to forecasting results, model comparisons, exploratory analysis, feature explanations, uncertainty intervals, and scenario simulation.

**API service:** https://aircast-research.onrender.com

**Interactive API documentation:** https://aircast-research.onrender.com/docs

**API health check:** https://aircast-research.onrender.com/health

*Note: Render free services may take some time to respond after periods of inactivity.*

---

## 1. Project Overview

AirCast Research evaluates direct multi-horizon PM2.5 forecasting using 420,768 hourly observations from 12 monitoring stations in Beijing, China.

The dataset covers March 1, 2013, to February 28, 2017.

The framework predicts PM2.5 concentrations at three forecasting horizons:

| Forecast Horizon | Forecasting Focus | Recommended Model |
|---|---|---|
| +1 hour | Short-term particulate dynamics | XGBoost |
| +6 hours | Intermediate atmospheric changes | Pruned XGBoost |
| +24 hours | Day-ahead air-quality forecasting | Fast XGBoost |

The system uses 47 engineered input features covering pollutant measurements, historical particulate concentrations, rolling statistics, meteorological variables, wind components, and temporal information.

### Main Objectives

- Develop accurate multi-horizon PM2.5 forecasting models.
- Prevent temporal leakage through chronological data splitting and boundary purging.
- Explain model predictions using native XGBoost TreeSHAP.
- Quantify forecast uncertainty using normalized split conformal prediction.
- Evaluate generalization across different monitoring stations.
- Improve inference efficiency and model storage.
- Provide interactive model inspection through a web dashboard and API.

### Forecasting Timeline

```text
Historical Input Data                         Future Predictions

t-48h          t-1h           t
  |-------------|-------------|
  <--- 47 causal input features --->

                              |------ +1h
                              |-------------- +6h
                              |---------------------------- +24h
```

All prediction features are designed to use information available at or before the forecast origin.

---

## 2. Key Forecasting Results

Three recommended XGBoost models were selected based on forecasting performance, model complexity, storage requirements, and inference efficiency.

### Recommended Model Performance

| Horizon | Model Artifact | Test Samples | MAE (µg/m³) | RMSE (µg/m³) | R² |
|---|---|---:|---:|---:|---:|
| +1 hour | xgboost_h1.joblib | 42,554 | 11.5723 | 23.3344 | 0.9499 |
| +6 hours | xgboost_pruned_h6.joblib | 42,494 | 37.0350 | 62.6263 | 0.6389 |
| +24 hours | xgboost_fast_h24.joblib | 42,278 | 66.0738 | 92.5722 | 0.2121 |

### Model Configurations

| Horizon | Number of Trees | Maximum Depth | Model Size |
|---|---:|---:|---:|
| +1 hour | 180 | 7 | 1.158 MB |
| +6 hours | 90 | 6 | 0.433 MB |
| +24 hours | 50 | 5 | 0.136 MB |

### Interpretation

**+1 Hour Forecasting**

The model achieves an R² of 0.9499. Short-term particulate concentrations are strongly influenced by recent pollution levels and autoregressive patterns.

**+6 Hour Forecasting**

The pruned model achieves an R² of 0.6389 while reducing the number of trees from 180 to 90. Its MAE is approximately 0.24% higher than the unpruned benchmark.

**+24 Hour Forecasting**

Day-ahead forecasting is more challenging because atmospheric conditions and pollution transport can change substantially. The selected 50-tree model achieves an MAE of 66.0738 µg/m³ and outperforms the persistence baseline by 12.25 µg/m³ in MAE.

The lower R² at this horizon reflects the greater uncertainty associated with longer-range forecasting.

---

## 3. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Core implementation |
| Forecasting | XGBoost | Multi-horizon PM2.5 prediction |
| Classical ML | Scikit-learn | Baselines and evaluation |
| Deep Learning | PyTorch | LSTM benchmark experiments |
| Data Processing | Pandas, NumPy | Data preparation and feature engineering |
| Storage | Parquet, PyArrow | Processed feature storage |
| Explainability | Native XGBoost TreeSHAP | Feature attribution |
| Uncertainty | Split Conformal Prediction | Prediction intervals |
| Backend | FastAPI, Uvicorn | REST API and inference |
| Dashboard | Streamlit | Interactive research interface |
| Testing | Pytest | Automated regression and validation |

The documented research benchmark environment uses Python 3.13.5 and CPU-based execution. The deployed services use their own Render runtime environments.

---

## 4. Dataset and Preprocessing

**Dataset:** Beijing Multi-Site Air-Quality Dataset (PRSA 2017)

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)

### Dataset Characteristics

| Property | Description |
|---|---|
| Total observations | 420,768 |
| Monitoring stations | 12 |
| Time resolution | Hourly |
| Time period | March 2013 – February 2017 |
| Target variable | PM2.5 |
| Forecast horizons | 1, 6, and 24 hours |
| Engineered features | 47 |

The dataset includes the following air-quality measurements:

- PM2.5
- PM10
- SO2
- NO2
- CO
- O3

Meteorological variables include temperature, pressure, dew point, rainfall, wind speed, and wind direction.

### Data Preparation Safeguards

- Future target values are not imputed.
- Missing target observations are excluded from evaluation.
- Input features are constructed using historical information.
- Station-aware preprocessing is used.
- Chronological partitions are used for model evaluation.
- Boundary embargoes reduce temporal leakage.

A total of 8,739 unobserved target rows are excluded rather than synthetically reconstructed.

### Feature Engineering

The 47 engineered features include:

1. Temporal cycles and calendar indicators.
2. Pollutant measurements.
3. Historical PM2.5 lag features.
4. Rolling means, standard deviations, minima, and maxima.
5. Meteorological measurements.
6. Wind direction transformations and wind-vector components.

---

## 5. Chronological Evaluation Methodology

AirCast uses a chronological purged evaluation strategy to reduce information leakage between training, validation, and testing.

| Partition | Period | Approximate Samples |
|---|---|---:|
| Training | 2013-03-01 to 2015-12-31 | 291,574 |
| Embargo | 24 hours | Purged |
| Validation | 2016-01-02 to 2016-09-30 | 77,316 |
| Embargo | 24 hours | Purged |
| Testing | 2016-10-02 to 2017-02-28 | 42,278–42,554 |

The final test partition is kept separate from model selection and calibration.

### Conformal Calibration Separation

The validation period is further separated into:

- Early validation: model exploration and tuning.
- Calibration Split A: estimation of the dispersion parameters.
- Calibration Split B: computation of conformal non-conformity scores.
- Final test partition: independent evaluation of empirical coverage.

This separation is intended to prevent calibration data from being reused for final performance evaluation.

---

## 6. Multi-Model Benchmarking

The project compares seven model families across three forecasting horizons.

| Model | Role |
|---|---|
| Persistence | Simple forecasting baseline |
| Linear Regression | Linear reference model |
| Ridge Regression | Regularized linear model |
| Random Forest | Tree ensemble baseline |
| HistGradientBoosting | Gradient boosting baseline |
| XGBoost | Recommended forecasting family |
| PyTorch LSTM | Sequential deep-learning benchmark |

### Important Benchmark Findings

- Short-term forecasting is strongly influenced by recent PM2.5 values.
- Intermediate forecasting benefits from nonlinear tree-based learning.
- Longer forecasting horizons are more difficult because atmospheric conditions change over time.
- The recommended XGBoost models provide a practical balance between accuracy, storage, and inference cost.

The detailed benchmark results are documented in the repository's research reports and metric artifacts.

---

## 7. Explainable AI Using TreeSHAP

AirCast uses native XGBoost TreeSHAP through:

```python
xgb.Booster.predict(pred_contribs=True)
```

This provides feature-level contributions to individual model predictions.

### Important Features

| Feature | Mean Absolute SHAP | Interpretation |
|---|---:|---|
| PM2.5 (t-0) | 32.41 | Recent particulate concentration |
| PM2.5 rolling mean (6h) | 14.85 | Recent pollution trend |
| wind_v | 11.20 | Meridional wind component |
| dew_point_depression | 9.45 | Moisture-related atmospheric conditions |
| WSPM | 7.80 | Wind speed |
| is_heating_season | 6.92 | Seasonal heating indicator |
| NO2 | 5.10 | Combustion-related pollutant |
| TEMP | 4.30 | Temperature |

### Wind Vector Transformation

Wind direction and wind speed are converted into two components:

\[
u=-WSPM\sin(\theta)
\]

\[
v=-WSPM\cos(\theta)
\]

These components represent zonal and meridional wind movement.

**Scientific limitation:** SHAP values describe learned model associations. They do not establish physical causation or prove atmospheric mechanisms.

---

## 8. Uncertainty Quantification

AirCast implements normalized split conformal prediction to estimate prediction intervals around its point forecasts.

The method uses calibration residuals scaled by an estimated conditional dispersion function.

\[
s_i=\frac{|y_i-\hat{y}_i|}{\hat{\sigma}(\hat{y}_i)}
\]

The dispersion estimate is:

\[
\hat{\sigma}(\hat{y})=
\max(5.0,\beta_0+\beta_1\hat{y})
\]

The resulting prediction interval is:

\[
C(x)=
\left[
\max(0,\hat{y}-q\hat{\sigma}),
\hat{y}+q\hat{\sigma}
\right]
\]

### Empirical Test Coverage

The nominal target confidence is 90%.

| Horizon | Test Coverage | Mean Interval Width |
|---|---:|---:|
| +1 hour | 86.36% | 38.97 µg/m³ |
| +6 hours | 83.04% | 120.26 µg/m³ |
| +24 hours | 70.17% | 163.16 µg/m³ |
| Pooled | 79.88% | 107.28 µg/m³ |

### Severe Pollution Coverage

For severe pollution observations above 250 µg/m³:

| Horizon | Empirical Coverage |
|---|---:|
| +1 hour | 81.66% |
| +6 hours | 65.97% |
| +24 hours | 23.80% |

**Important limitation:** The nominal 90% target is not achieved across the complete test distribution. Conformal prediction's usual coverage assumptions depend on exchangeability, and seasonal or severe distribution shifts can reduce empirical coverage substantially. AirCast does not claim guaranteed 90% coverage under non-stationary atmospheric conditions.

---

## 9. Spatial Generalization

AirCast evaluates spatial transferability through 12-fold Leave-One-Station-Out (LOSO) evaluation.

In each fold:

- Eleven monitoring stations are used for model training.
- One station is held out for evaluation.
- The held-out station retains access to its own historical telemetry.

### Results

| Metric | Macro-Averaged Result |
|---|---:|
| Test samples | 411,165 |
| MAE | 46.33 ± 1.65 µg/m³ |
| RMSE | 64.48 µg/m³ |
| R² | 0.3550 |

This experiment evaluates transfer to a new monitoring station when its local historical measurements are available.

It does not represent zero-shot prediction at completely uninstrumented geographic locations.

---

## 10. Robustness and Sensor Fault Testing

The framework evaluates how forecasting performance changes when sensor measurements are corrupted or unavailable.

### Sensor Noise

The +24-hour forecasting model was evaluated under Gaussian noise injected into raw sensor measurements.

| Noise Level | Mean Test MAE |
|---|---:|
| 5% | 66.67 ± 0.02 µg/m³ |
| 15% | 66.98 ± 0.03 µg/m³ |
| 25% | 67.49 ± 0.07 µg/m³ |

### Sensor Failure Experiments

| Failure Scenario | MAE Change |
|---|---:|
| CO sensor failure | -0.01 µg/m³ |
| Precursor gas suite failure | +0.30 µg/m³ |
| Wind sensor suite failure | +2.06 µg/m³ |
| Particulate sensor disconnection | +5.68 µg/m³ |

These experiments help identify which sensor groups have the greatest influence on forecasting performance.

---

## 11. Computational Efficiency

The project includes CPU optimization experiments covering model size, memory usage, feature extraction, and inference.

### Optimization Results

| Optimization | Baseline | Optimized |
|---|---:|---:|
| Active DataFrame memory | 227.38 MB | 121.45 MB |
| Feature extraction | 4.201 s | 0.0496 s |
| Raw XGBoost C++ inference mean | 35.60 ms | 4.33 ms |
| Batch throughput | 286.6 samples/s | 188,688.3 samples/s |

### Inference Latency Boundaries

Different measurements represent different stages of inference and should not be confused.

| Measurement | Result |
|---|---:|
| Raw XGBoost C++ inference P95 | 7.43 ms |
| In-process service P95 | 43.32–44.13 ms |
| REST HTTP end-to-end P95 | 99.66 ms |

The raw model inference measurement meets the 15 ms target.

The in-process service and REST HTTP measurements do not meet that target.

All these measurements are CPU research benchmarks, not guaranteed cloud deployment response times.

---

## 12. System Architecture

```text
             Beijing Air Quality Dataset
                         |
                         v
              Data Loading & Cleaning
                         |
                         v
              Causal Feature Engineering
                         |
                         v
                 47 Input Features
                         |
                         v
                Chronological Splits
                         |
                         v
                XGBoost Forecasting
                  /      |      \
                 /       |       \
              +1h       +6h      +24h
                 \       |       /
                  \      |      /
                         v
              Conformal Calibration
                         |
                         v
                Prediction Intervals
                         |
              +----------+----------+
              |                     |
              v                     v
          FastAPI API          Streamlit Dashboard
              |                     |
              v                     v
       Prediction Service      Visual Analysis
```

### API Capabilities

The FastAPI application provides:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service health |
| `/model-info` | GET | Model metadata |
| `/features` | GET | Feature schema |
| `/metrics` | GET | Benchmark metrics |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch prediction |

The API supports forecasting horizons of 1, 6, and 24 hours.

Prediction requests require the complete 47-feature input schema. Missing or non-finite features and unsupported horizons are rejected through input validation.

### Streamlit Dashboard

The dashboard provides interactive research panels covering:

1. Executive Overview
2. Exploratory Data Analysis
3. Multi-Horizon Benchmarking
4. Feature Engineering and Ablation
5. Temporal and Seasonal Analysis
6. Spatial Generalization
7. Conformal Uncertainty Calibration
8. TreeSHAP Explainability
9. Synthetic Scenario Simulation

---

## 13. Deployment

AirCast Research provides two deployed services:

**Interactive Dashboard**

https://aircast-research-1.onrender.com/

**FastAPI Backend**

https://aircast-research.onrender.com/

The dashboard provides the interactive research interface, while the API provides model inference and metadata endpoints.

The deployed services are intended for research demonstration and interactive exploration. They are not hardened multi-tenant production systems.

### Deployment Verification

The API has been tested for:

- Health endpoint response.
- Model artifact loading.
- Supported forecasting horizons.
- Feature schema retrieval.
- Single-record prediction.
- Calibrated prediction interval generation.

The dashboard has also been deployed and its interactive interface verified.

---

## 14. Installation and Local Execution

### Requirements

- Python 3.13.5 for the documented research environment.
- Git.
- The repository's required Python dependencies.

### Clone the Repository

```bash
git clone https://github.com/sreeja1547/aircast-research.git

cd aircast-research
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### Launch the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

http://localhost:8000/docs

### Launch the Dashboard

In another terminal, activate the same virtual environment and run:

```bash
streamlit run dashboard/app.py
```

Open:

http://localhost:8501

---

## 15. Dataset Setup

The processed feature cache is included in:

```text
data/processed/forecasting_features_cache.parquet
```

If you want to regenerate the feature cache from the original dataset:

1. Download the Beijing Multi-Site Air-Quality Dataset from the UCI repository.
2. Extract the 12 station CSV files into:

```text
PRSA_Data_20130301-20170228/
```

3. Run:

```bash
python scripts/prepare_processed_data.py
```

The raw station CSV files are excluded from Git through `.gitignore`.

---

## 16. Testing

The repository includes automated tests covering:

- API behavior and validation.
- Model inference.
- Feature engineering.
- Preprocessing.
- Conformal uncertainty.
- Explainability.
- Forecast evaluation.
- Historical dashboard replay.
- Production import behavior.
- Robustness and methodology audits.

### Run Tests

```bash
pytest tests/ -v
```

**Latest verified local test result:**

```text
76 passed, 0 failed, 4 warnings
```

Test results describe the verified local code version and may change after subsequent modifications.

---

## 17. Repository Structure

```text
aircast-research/
│
├── api/
│   ├── main.py
│   ├── services.py
│   └── schemas.py
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── uncertainty/
│   ├── explainability/
│   ├── experiments/
│   └── evaluation/
│
├── data/
│   └── processed/
│       └── forecasting_features_cache.parquet
│
├── artifacts/
│   ├── models/
│   ├── metrics/
│   └── plots/
│
├── research/
│
├── scripts/
│
├── tests/
│
├── requirements.txt
├── requirements-prod.txt
├── render.yaml
├── .gitignore
├── LICENSE
└── README.md
```

### Model Artifacts

The recommended forecasting models and their conformal calibrators are included in `artifacts/models/`.

The three recommended forecasting models are:

- `xgboost_h1.joblib`
- `xgboost_pruned_h6.joblib`
- `xgboost_fast_h24.joblib`

The corresponding conformal calibration artifacts are also maintained in the repository.

Additional experimental baseline models are preserved locally and excluded from Git to maintain a compact repository.

---

## 18. Research Documentation

The `research/` directory contains detailed technical documentation covering:

- Exploratory data analysis.
- Dataset preparation and causal preprocessing.
- Feature engineering methodology.
- Model benchmarking and hyperparameter tuning.
- Explainable AI and TreeSHAP.
- Conformal uncertainty quantification.
- Temporal and spatial generalization.
- Robustness and sensor stress testing.
- Computational efficiency.
- Deployment and release verification.

The `artifacts/metrics/` and `artifacts/plots/` directories contain supporting benchmark results and research visualizations.

---

## 19. Citation and License

If you use this project in academic or applied research, please cite it as software:

```bibtex
@software{yenduri_aircast_research_2026,
  author = {Yenduri Sreeja},
  title = {AirCast Research: A Research-Driven Framework for
           Explainable, Uncertainty-Aware, and Generalizable
           Multi-Horizon Air Quality Forecasting},
  year = {2026},
  url = {https://github.com/sreeja1547/aircast-research}
}
```

### Dataset Attribution

The project uses the Beijing Multi-Site Air-Quality Dataset (PRSA 2017), distributed through the UCI Machine Learning Repository.

Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). Cautionary tales on air-quality improvement in Beijing. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 473(2205), 20170457.

The original dataset remains subject to its own attribution and usage terms.

### License

This repository is licensed under the MIT License.

Copyright (c) 2026 Yenduri Sreeja.

See [LICENSE](LICENSE) for the complete license text.

---

## 20. Disclaimer

AirCast Research is an applied machine learning research and demonstration project.

Its forecasts and uncertainty intervals are intended for research and educational exploration. They should not be used as the sole basis for regulatory decisions, public-health interventions, or life-critical applications.

The reported model performance is based on historical Beijing monitoring-station data and does not guarantee equivalent performance in other locations or future atmospheric conditions.