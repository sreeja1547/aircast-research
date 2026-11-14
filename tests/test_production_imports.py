"""Automated tests for Phase 1 lazy PyTorch imports and backward compatibility."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def run_isolated_code(code: str) -> subprocess.CompletedProcess:
    """Run Python snippet in a clean, isolated subprocess with REPO_ROOT in sys.path."""
    full_code = f"import sys\nsys.path.insert(0, {repr(REPO_ROOT)})\n{code}"
    return subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=True,
        text=True,
        check=True,
    )


def test_api_services_import_does_not_load_torch():
    """Verify that importing api.services does not load PyTorch into sys.modules."""
    code = """
import api.services
assert 'torch' not in sys.modules, f"torch unexpectedly imported: {list(sys.modules.keys())}"
print("OK: api.services imported without torch")
"""
    res = run_isolated_code(code)
    assert "OK: api.services imported without torch" in res.stdout


def test_xgboost_inference_import_does_not_load_torch():
    """Verify that importing XGBoostForecaster and registry does not load PyTorch."""
    code = """
from src.models import XGBoostForecaster, list_available_models, get_model
assert 'torch' not in sys.modules, "torch imported during XGBoost/registry import"
models = list_available_models()
assert 'xgboost' in models
assert 'lstm' in models
assert 'torch' not in sys.modules, "torch imported during list_available_models"
xgb = get_model('xgboost')
assert isinstance(xgb, XGBoostForecaster)
assert 'torch' not in sys.modules, "torch imported during get_model('xgboost')"
print("OK: XGBoost inference without torch")
"""
    res = run_isolated_code(code)
    assert "OK: XGBoost inference without torch" in res.stdout


def test_lazy_lstm_import_works_backward_compatible():
    """Verify that from src.models import LSTMForecaster works via PEP 562 lazy loading."""
    code = """
assert 'torch' not in sys.modules
from src.models import LSTMForecaster
assert 'torch' in sys.modules, "torch should be loaded after importing LSTMForecaster"
assert LSTMForecaster.__name__ == 'LSTMForecaster'
model = LSTMForecaster()
assert model.model_type == 'deep_learning'
print("OK: LSTMForecaster lazy import works")
"""
    res = run_isolated_code(code)
    assert "OK: LSTMForecaster lazy import works" in res.stdout


def test_registry_get_model_lstm_lazy_loads_torch():
    """Verify that get_model('lstm') lazily loads PyTorch only when requested."""
    code = """
from src.models import get_model
assert 'torch' not in sys.modules
lstm = get_model('lstm')
assert 'torch' in sys.modules
assert lstm.name == 'lstm'
assert lstm.model_type == 'deep_learning'
print("OK: get_model('lstm') lazily instantiates deep learning model")
"""
    res = run_isolated_code(code)
    assert "OK: get_model('lstm') lazily instantiates deep learning model" in res.stdout


def test_all_existing_model_imports_backward_compatible():
    """Verify all 8 exports from src.models continue to import normally."""
    code = """
from src.models import (
    BaseForecaster,
    PersistenceForecaster,
    LinearRegressionForecaster,
    RidgeForecaster,
    RandomForestForecaster,
    HistGradientBoostingForecaster,
    XGBoostForecaster,
    LSTMForecaster,
    MODEL_REGISTRY,
    get_model,
    list_available_models,
)
for name in list_available_models():
    assert name in MODEL_REGISTRY
print("OK: All models backward compatible")
"""
    res = run_isolated_code(code)
    assert "OK: All models backward compatible" in res.stdout
