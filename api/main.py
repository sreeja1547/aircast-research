"""FastAPI application for AirCast Research.

Exposes RESTful endpoints for health checks, model catalog, feature registry,
real-time multi-horizon PM2.5 forecasting, and batch inference.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureRegistryResponse,
    HealthResponse,
    ModelCatalogResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from api.services import AirCastInferenceService
from src.features.registry import FEATURE_GROUPS, get_feature_group

app = FastAPI(
    title="AirCast Research API",
    description="A Research-Driven Framework for Explainable, Uncertainty-Aware and Generalizable Multi-Horizon Air Quality Forecasting",
    version="1.0.0",
)

# Restrict CORS origins to local development hosts
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8501",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

service = AirCastInferenceService()


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """Return API health status, version, and active forecast horizons."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available_horizons": [1, 6, 24],
    }


@app.get("/model-info", response_model=ModelCatalogResponse, tags=["Metadata"])
def get_model_info():
    """Return metadata about verified production forecasting models and supported lead times."""
    horizons = [1, 6, 24]
    active_models = [
        ModelInfo(
            model_name="recommended",
            model_type="tree",
            horizon_hours=1,
            n_features=47,
            is_loaded=(Path("artifacts/models/xgboost_h1.joblib").exists()),
            artifact_file="xgboost_h1.joblib",
        ),
        ModelInfo(
            model_name="recommended",
            model_type="tree",
            horizon_hours=6,
            n_features=47,
            is_loaded=(Path("artifacts/models/xgboost_pruned_h6.joblib").exists()),
            artifact_file="xgboost_pruned_h6.joblib",
        ),
        ModelInfo(
            model_name="recommended",
            model_type="tree",
            horizon_hours=24,
            n_features=47,
            is_loaded=(Path("artifacts/models/xgboost_fast_h24.joblib").exists()),
            artifact_file="xgboost_fast_h24.joblib",
        ),
    ]
    for h in horizons:
        active_models.append(
            ModelInfo(
                model_name="persistence",
                model_type="baseline",
                horizon_hours=h,
                n_features=47,
                is_loaded=True,
                artifact_file="none (in-memory heuristic)",
            )
        )

    return {
        "active_models": active_models,
        "feature_count": 47,
        "supported_horizons": horizons,
        "default_model": "recommended",
    }


@app.get("/features", response_model=FeatureRegistryResponse, tags=["Features"])
def get_features():
    """Return all registered feature groups and column names."""
    groups = {k: get_feature_group(k) for k in FEATURE_GROUPS.keys()}
    groups["combined_all"] = get_feature_group("combined_all")
    return {
        "feature_groups": groups,
        "total_features_combined": len(groups["combined_all"]),
    }


@app.get("/metrics", tags=["Diagnostics"])
def get_benchmarking_metrics():
    """Return precomputed empirical benchmarking metrics from research experiments."""
    metrics_path = Path("artifacts/metrics/experiment1_benchmarking.json")
    if not metrics_path.exists():
        return {"status": "Metrics generation in progress. Run experiments to populate."}

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(req: PredictionRequest):
    """Generate multi-horizon point forecast and conformal prediction interval."""
    try:
        point_pred, lb, ub, latency_ms, interval_method, is_calibrated = service.predict_single(
            features=req.features,
            horizon=req.horizon_hours,
            model_name=req.model_name or "recommended",
            compute_interval=req.compute_interval,
            nominal_coverage=req.nominal_coverage,
        )
        resolved_artifact = service.get_resolved_artifact(req.model_name or "recommended", req.horizon_hours)

        return {
            "model_name": req.model_name or "recommended",
            "resolved_artifact": resolved_artifact,
            "horizon_hours": req.horizon_hours,
            "predicted_pm25_ug_m3": point_pred,
            "unit": "ug/m^3",
            "lower_bound": lb,
            "upper_bound": ub,
            "confidence_level": req.nominal_coverage if req.compute_interval else None,
            "interval_method": interval_method,
            "is_calibrated": is_calibrated,
            "inference_latency_ms": latency_ms,
        }
    except HTTPException:
        raise
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf),
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(req: BatchPredictionRequest):
    """Generate multi-sample batch predictions and prediction intervals."""
    if not req.samples:
        raise HTTPException(status_code=400, detail="Sample list cannot be empty.")

    try:
        preds, lbs, ubs, tot_lat, per_samp, interval_method, is_calibrated = service.predict_batch(
            samples=req.samples,
            horizon=req.horizon_hours,
            model_name=req.model_name or "recommended",
            compute_intervals=req.compute_intervals,
            nominal_coverage=req.nominal_coverage,
        )
        resolved_artifact = service.get_resolved_artifact(req.model_name or "recommended", req.horizon_hours)

        return {
            "model_name": req.model_name or "recommended",
            "resolved_artifact": resolved_artifact,
            "horizon_hours": req.horizon_hours,
            "sample_count": len(req.samples),
            "predictions": preds,
            "lower_bounds": lbs,
            "upper_bounds": ubs,
            "confidence_level": req.nominal_coverage if req.compute_intervals else None,
            "interval_method": interval_method,
            "is_calibrated": is_calibrated,
            "total_latency_ms": tot_lat,
            "per_sample_latency_ms": per_samp,
        }
    except HTTPException:
        raise
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf),
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}",
        )
