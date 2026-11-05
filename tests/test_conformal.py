"""Unit tests for Conformal Uncertainty Quantification.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import PersistenceForecaster
from src.uncertainty.conformal import ConformalPredictor, evaluate_conformal_coverage


def test_evaluate_conformal_coverage():
    """Verify empirical coverage calculation and sharpness metrics."""
    y_true = np.array([50.0, 75.0, 100.0, 160.0])
    lb = np.array([40.0, 70.0, 90.0, 150.0])
    ub = np.array([60.0, 80.0, 110.0, 170.0])

    res = evaluate_conformal_coverage(y_true, lb, ub, nominal_coverage=0.90)
    assert res["nominal_coverage"] == 0.90
    assert res["empirical_coverage"] == 1.0  # All points fall inside [lb, ub]
    assert res["mean_interval_width"] == 17.5
    assert res["extreme_sample_count"] == 1
    assert res["extreme_coverage"] == 1.0


def test_conformal_predictor_calibration():
    """Verify ConformalPredictor calibration and prediction interval bounds."""
    np.random.seed(42)
    X = pd.DataFrame({"PM2.5": np.random.uniform(20.0, 100.0, 100)})
    y = X["PM2.5"] + np.random.normal(0, 5.0, 100)

    base_model = PersistenceForecaster(target_col="PM2.5")
    base_model.fit(X.iloc[:50], y.iloc[:50])

    conformal = ConformalPredictor(base_model=base_model, default_coverage=0.90)
    conformal.calibrate(X.iloc[50:], y.iloc[50:])
    assert conformal.is_calibrated
    assert 0.90 in conformal.quantiles_
    assert conformal.quantiles_[0.90] > 0.0

    test_X = pd.DataFrame({"PM2.5": [50.0, 80.0]})
    intervals = conformal.predict_intervals(test_X, coverage_level=0.90)
    assert len(intervals) == 2
    assert "point_pred" in intervals.columns
    assert "lower_bound" in intervals.columns
    assert "upper_bound" in intervals.columns
    assert np.all(intervals["lower_bound"] >= 0.0)
    assert np.all(intervals["upper_bound"] >= intervals["lower_bound"])


def test_normalized_conformal_predictor():
    """Verify adaptive normalized ConformalPredictor with difficulty score."""
    np.random.seed(42)
    X = pd.DataFrame({"PM2.5": np.random.uniform(20.0, 200.0, 100)})
    # Heteroscedastic error: standard deviation scales with concentration
    y = X["PM2.5"] + np.random.normal(0, 0.1 * X["PM2.5"], 100)

    base_model = PersistenceForecaster(target_col="PM2.5")
    base_model.fit(X.iloc[:50], y.iloc[:50])

    norm_conformal = ConformalPredictor(base_model=base_model, default_coverage=0.90, method="normalized")
    norm_conformal.calibrate(X.iloc[50:], y.iloc[50:])
    assert norm_conformal.is_calibrated
    assert norm_conformal.method == "normalized"
    assert norm_conformal.dispersion_params_["beta_1"] >= 0.0

    # Low vs High pollution query: interval width should adapt
    query_df = pd.DataFrame({"PM2.5": [25.0, 250.0]})
    intervals = norm_conformal.predict_intervals(query_df, coverage_level=0.90)
    assert len(intervals) == 2
    # High pollution interval must be wider than low pollution interval due to adaptive scaling
    assert intervals["interval_width"].iloc[1] > intervals["interval_width"].iloc[0]
