"""Unit tests for TreeSHAP Explainability Engine.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_engine import ShapExplanationEngine
from src.models import XGBoostForecaster


def test_shap_explanation_engine(tmp_path):
    """Verify TreeSHAP explanation properties, additive efficiency, and plot generation."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "PM2.5": np.random.uniform(10.0, 150.0, n),
        "WSPM": np.random.uniform(0.5, 6.0, n),
        "TEMP": np.random.uniform(0.0, 30.0, n),
    })
    y = 0.8 * df["PM2.5"] - 4.0 * df["WSPM"] + np.random.normal(0, 3, n)

    model = XGBoostForecaster(n_estimators=30, max_depth=4, random_state=42)
    model.fit(df, y)

    engine = ShapExplanationEngine(model)
    res = engine.explain(df.iloc[:15])

    shap_vals = res["shap_values"]
    base_val = res["base_value"]
    mean_abs = res["mean_abs_shap"]

    assert shap_vals.shape == (15, 3)
    assert isinstance(base_val, float)
    assert len(mean_abs) == 3
    # Most important feature should be PM2.5 or WSPM
    assert mean_abs.index[0] in ["PM2.5", "WSPM"]

    # Verify additive efficiency: sum(phi_i) + base_val = raw prediction
    raw_preds = model.model.predict(df.iloc[:15])
    reconstructed = np.sum(shap_vals, axis=1) + base_val
    np.testing.assert_allclose(raw_preds, reconstructed, rtol=1e-4)

    # Test plot generation
    plot_file = tmp_path / "shap_test_plot.png"
    engine.generate_summary_plot(df.iloc[:20], output_path=str(plot_file), top_k=3)
    assert plot_file.exists()
    assert plot_file.stat().st_size > 1000
