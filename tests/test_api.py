"""Unit tests for the AirCast Research FastAPI application.
"""

from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health check response structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "available_horizons" in data
    assert 24 in data["available_horizons"]


def test_model_info_endpoint():
    """Verify model metadata catalog endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "active_models" in data
    assert data["feature_count"] == 47
    assert data["default_model"] == "xgboost"


def test_features_endpoint():
    """Verify feature registry endpoint."""
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert "combined_all" in data["feature_groups"]
    assert data["total_features_combined"] == 47


def test_metrics_endpoint():
    """Verify metrics endpoint returns 200."""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_predict_single_endpoint():
    """Verify single prediction with heuristic fallback for uncalibrated model."""
    sample_payload = {
        "horizon_hours": 24,
        "model_name": "persistence",
        "features": {
            "PM2.5": 78.5,
            "PM10": 110.0,
            "TEMP": 15.0,
            "WSPM": 2.5,
        },
        "compute_interval": True,
        "nominal_coverage": 0.90,
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["horizon_hours"] == 24
    assert data["predicted_pm25_ug_m3"] >= 0.0
    assert data["lower_bound"] is not None
    assert data["upper_bound"] >= data["lower_bound"]
    assert data["inference_latency_ms"] > 0.0
    # Baseline model does not have calibrated conformal artifacts; must be marked heuristic
    assert data["is_calibrated"] is False
    assert data["interval_method"] == "heuristic_fallback"


def test_predict_single_calibrated_conformal():
    """Verify single prediction uses horizon-specific calibrated conformal intervals."""
    for h in [1, 6, 24]:
        sample_payload = {
            "horizon_hours": h,
            "model_name": "xgboost",
            "features": {
                "PM2.5": 95.0,
                "PM10": 140.0,
                "TEMP": 8.0,
                "PRES": 1018.0,
                "DEWP": -4.0,
                "WSPM": 1.8,
            },
            "compute_interval": True,
            "nominal_coverage": 0.90,
        }
        response = client.post("/predict", json=sample_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["horizon_hours"] == h
        assert data["predicted_pm25_ug_m3"] >= 0.0
        assert data["lower_bound"] is not None
        assert data["upper_bound"] > data["lower_bound"]
        assert data["is_calibrated"] is True
        assert data["interval_method"] == "calibrated_normalized_conformal"


def test_predict_no_intervals():
    """Verify endpoint behavior when compute_interval is False."""
    payload = {
        "horizon_hours": 1,
        "model_name": "xgboost",
        "features": {"PM2.5": 50.0},
        "compute_interval": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["lower_bound"] is None
    assert data["upper_bound"] is None
    assert data["interval_method"] is None
    assert data["is_calibrated"] is False


def test_predict_batch_endpoint():
    """Verify batch prediction endpoint with heuristic fallback for persistence."""
    batch_payload = {
        "horizon_hours": 1,
        "model_name": "persistence",
        "samples": [
            {"PM2.5": 45.0, "WSPM": 3.0},
            {"PM2.5": 120.0, "WSPM": 1.0},
            {"PM2.5": 210.0, "WSPM": 0.5},
        ],
        "compute_intervals": True,
        "nominal_coverage": 0.95,
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sample_count"] == 3
    assert len(data["predictions"]) == 3
    assert len(data["lower_bounds"]) == 3
    assert len(data["upper_bounds"]) == 3
    assert data["is_calibrated"] is False
    assert data["interval_method"] == "heuristic_fallback"


def test_predict_batch_calibrated_conformal():
    """Verify batch prediction with calibrated conformal intervals."""
    batch_payload = {
        "horizon_hours": 6,
        "model_name": "xgboost",
        "samples": [
            {"PM2.5": 30.0, "WSPM": 4.0},
            {"PM2.5": 180.0, "WSPM": 0.8},
        ],
        "compute_intervals": True,
        "nominal_coverage": 0.90,
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sample_count"] == 2
    assert len(data["predictions"]) == 2
    assert data["is_calibrated"] is True
    assert data["interval_method"] == "calibrated_normalized_conformal"
    assert data["lower_bounds"][0] <= data["upper_bounds"][0]
    assert data["lower_bounds"][1] <= data["upper_bounds"][1]
