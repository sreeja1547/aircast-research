"""Inference service layer managing forecaster models, conformal intervals, and schema compliance.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.features.registry import get_feature_group
from src.models import (
    BaseForecaster,
    HistGradientBoostingForecaster,
    PersistenceForecaster,
    RidgeForecaster,
    XGBoostForecaster,
    get_model,
)
from src.uncertainty.conformal import ConformalPredictor


class AirCastInferenceService:
    """Manages active models, feature schema alignment, and real-time predictions."""

    def __init__(self, models_dir: str = "artifacts/models") -> None:
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, BaseForecaster] = {}
        self.conformal_calibrators: Dict[int, ConformalPredictor] = {}
        self.feature_names = get_feature_group("combined_all")
        self.feature_idx_map = {col: i for i, col in enumerate(self.feature_names)}

    def _fill_feature_defaults(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate presence of all 47 required features and align to canonical ordering.

        Strictly rejects missing or non-finite features with ValueError.
        Does not silently impute or invent missing values.
        """
        if not features:
            raise ValueError("Feature payload cannot be empty.")
        missing = [col for col in self.feature_names if col not in features or features[col] is None]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        validated = {}
        for col in self.feature_names:
            val = float(features[col])
            if not np.isfinite(val):
                raise ValueError(f"Feature column '{col}' has invalid non-finite value: {val}")
            validated[col] = val
        return validated

    def validate_and_align_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Public alias for feature schema validation and ordering."""
        return self._fill_feature_defaults(features)

    def _resolve_model_name(self, model_name: str, horizon: int) -> str:
        """Resolve aliases like 'recommended', 'xgboost', 'pareto', or 'optimized' to horizon-specific Pareto models."""
        clean = model_name.lower().strip()
        if clean in ("recommended", "xgboost", "xgb", "pareto", "optimized"):
            if horizon == 1:
                return "xgboost"
            elif horizon == 6:
                return "xgboost_pruned"
            elif horizon == 24:
                return "xgboost_fast"
        return clean

    def _get_model_key(self, model_name: str, horizon: int) -> str:
        resolved = self._resolve_model_name(model_name, horizon)
        return f"{resolved}_h{horizon}"

    def get_conformal_calibrator(self, horizon: int) -> Optional[ConformalPredictor]:
        """Retrieve horizon-specific calibrated ConformalPredictor instance."""
        if horizon in self.conformal_calibrators:
            return self.conformal_calibrators[horizon]

        calib_path = self.models_dir / f"conformal_calibrator_h{horizon}.joblib"
        if calib_path.exists():
            try:
                calibrator = ConformalPredictor.load(calib_path)
                self.conformal_calibrators[horizon] = calibrator
                return calibrator
            except Exception as e:
                print(f"[Warning] Failed loading conformal calibrator from {calib_path}: {e}")

        return None

    def get_model(self, model_name: str, horizon: int) -> BaseForecaster:
        """Retrieve model instance from cache or disk artifact.

        Raises FileNotFoundError if the requested model artifact is not on disk.
        Does not perform silent dummy training.
        """
        resolved = self._resolve_model_name(model_name, horizon)
        key = self._get_model_key(resolved, horizon)
        if key in self.loaded_models:
            return self.loaded_models[key]

        # Check disk artifact
        artifact_path = self.models_dir / f"{key}.joblib"
        if artifact_path.exists():
            try:
                model = joblib.load(artifact_path)
                self.loaded_models[key] = model
                return model
            except Exception as e:
                raise RuntimeError(f"Failed loading model artifact {artifact_path}: {e}")

        # Baseline unweighted persistence heuristic (stateless)
        clean_name = resolved.lower().strip()
        if clean_name == "persistence":
            model = PersistenceForecaster(name=f"persistence_h{horizon}", target_col="PM2.5")
            self.loaded_models[key] = model
            return model

        raise FileNotFoundError(
            f"Model artifact '{key}.joblib' was not found in '{self.models_dir}'. "
            f"Available tracked deployment artifacts are 'xgboost_h1.joblib', 'xgboost_pruned_h6.joblib', "
            f"'xgboost_fast_h24.joblib', or baseline 'persistence'."
        )

    def get_resolved_artifact(self, model_name: str, horizon: int) -> str:
        """Return the resolved artifact filename for a model alias and horizon."""
        resolved = self._resolve_model_name(model_name, horizon)
        if resolved == "persistence":
            return "persistence (in-memory)"
        return f"{self._get_model_key(resolved, horizon)}.joblib"

    def predict_single(
        self,
        features: Dict[str, float],
        horizon: int = 24,
        model_name: str = "recommended",
        compute_interval: bool = True,
        nominal_coverage: float = 0.90,
    ) -> Tuple[float, Optional[float], Optional[float], float, Optional[str], bool]:
        """Generate point prediction and conformal interval for a single input record."""
        t0 = time.perf_counter()
        resolved_name = self._resolve_model_name(model_name, horizon)
        key = self._get_model_key(resolved_name, horizon)
        resolved_artifact = f"{key}.joblib" if resolved_name != "persistence" else "persistence (in-memory)"
        model = self.get_model(resolved_name, horizon)

        # Build 1-row DataFrame aligned to schema
        row_dict = self._fill_feature_defaults(features)
        df_sample = pd.DataFrame([row_dict])

        preds = model.predict(df_sample)
        point_pred = float(preds[0])

        lower_bound = None
        upper_bound = None
        interval_method = None
        is_calibrated = False

        if compute_interval:
            clean_name = resolved_name.lower().strip()
            calibrator = self.get_conformal_calibrator(horizon)
            is_xgb_family = (
                clean_name in ("xgboost", "xgb", "xgboost_pruned", "xgboost_fast")
                or clean_name.startswith("xgboost")
                or clean_name in ("recommended", "pareto", "optimized")
            )

            # Check model identity binding: verify calibrator matches deployment model
            calibrator_matches_model = False
            if calibrator is not None and calibrator.is_calibrated and is_xgb_family:
                target_model_name = getattr(calibrator, "model_artifact_name", None)
                if target_model_name is None or target_model_name == f"{key}.joblib" or f"{clean_name}_h{horizon}" in str(target_model_name):
                    calibrator_matches_model = True

            if calibrator_matches_model and calibrator is not None:
                try:
                    q_val = calibrator.quantiles_.get(nominal_coverage)
                    if q_val is None:
                        q_val = calibrator._compute_quantile(nominal_coverage)

                    if calibrator.method == "normalized":
                        b0 = calibrator.dispersion_params_.get("beta_0", calibrator.sigma_min)
                        b1 = calibrator.dispersion_params_.get("beta_1", 0.0)
                        sigma_x = max(calibrator.sigma_min, b0 + b1 * point_pred)
                        margin = q_val * sigma_x
                    else:
                        margin = q_val

                    lower_bound = round(float(max(0.0, point_pred - margin)), 2)
                    upper_bound = round(float(point_pred + margin), 2)
                    interval_method = f"calibrated_{calibrator.method}_conformal"
                    is_calibrated = True
                except Exception as e:
                    print(f"[Warning] Calibrated conformal calculation failed: {e}")

            if not is_calibrated:
                margin_map = {1: 15.0, 6: 35.0, 24: 55.0}
                base_margin = margin_map.get(horizon, 45.0)
                z_factor = 1.645 if nominal_coverage <= 0.90 else 1.96
                margin = base_margin * (z_factor / 1.645)

                lower_bound = round(max(0.0, point_pred - margin), 2)
                upper_bound = round(point_pred + margin, 2)
                interval_method = "heuristic_fallback"
                is_calibrated = False

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return round(point_pred, 2), lower_bound, upper_bound, latency_ms, interval_method, is_calibrated

    def predict_batch(
        self,
        samples: List[Dict[str, float]],
        horizon: int = 24,
        model_name: str = "recommended",
        compute_intervals: bool = True,
        nominal_coverage: float = 0.90,
    ) -> Tuple[List[float], Optional[List[float]], Optional[List[float]], float, float, Optional[str], bool]:
        """Generate point predictions and intervals for a list of input records using vectorized buffers."""
        t0 = time.perf_counter()
        resolved_name = self._resolve_model_name(model_name, horizon)
        key = self._get_model_key(resolved_name, horizon)
        resolved_artifact = f"{key}.joblib" if resolved_name != "persistence" else "persistence (in-memory)"
        model = self.get_model(resolved_name, horizon)

        n_samples = len(samples)
        n_feats = len(self.feature_names)
        mat = np.empty((n_samples, n_feats), dtype=np.float32)

        for i, s in enumerate(samples):
            filled_s = self._fill_feature_defaults(s)
            for col, j in self.feature_idx_map.items():
                mat[i, j] = filled_s[col]

        df_batch = pd.DataFrame(mat, columns=self.feature_names)
        preds = model.predict(df_batch)
        point_preds = [round(float(p), 2) for p in preds]

        lower_bounds = None
        upper_bounds = None
        interval_method = None
        is_calibrated = False

        if compute_intervals:
            clean_name = resolved_name.lower().strip()
            calibrator = self.get_conformal_calibrator(horizon)
            is_xgb_family = (
                clean_name in ("xgboost", "xgb", "xgboost_pruned", "xgboost_fast")
                or clean_name.startswith("xgboost")
                or clean_name in ("recommended", "pareto", "optimized")
            )

            calibrator_matches_model = False
            if calibrator is not None and calibrator.is_calibrated and is_xgb_family:
                target_model_name = getattr(calibrator, "model_artifact_name", None)
                if target_model_name is None or target_model_name == f"{key}.joblib" or f"{clean_name}_h{horizon}" in str(target_model_name):
                    calibrator_matches_model = True

            if calibrator_matches_model and calibrator is not None:
                try:
                    q_val = calibrator.quantiles_.get(nominal_coverage)
                    if q_val is None:
                        q_val = calibrator._compute_quantile(nominal_coverage)

                    preds_arr = np.asarray(point_preds, dtype=np.float32)
                    if calibrator.method == "normalized":
                        b0 = calibrator.dispersion_params_.get("beta_0", calibrator.sigma_min)
                        b1 = calibrator.dispersion_params_.get("beta_1", 0.0)
                        sigma_x = np.maximum(calibrator.sigma_min, b0 + b1 * preds_arr)
                        margins = q_val * sigma_x
                    else:
                        margins = np.full_like(preds_arr, q_val)

                    lower_bounds = [round(float(max(0.0, p - m)), 2) for p, m in zip(point_preds, margins)]
                    upper_bounds = [round(float(p + m), 2) for p, m in zip(point_preds, margins)]
                    interval_method = f"calibrated_{calibrator.method}_conformal"
                    is_calibrated = True
                except Exception as e:
                    print(f"[Warning] Calibrated batch conformal calculation failed: {e}")

            if not is_calibrated:
                margin_map = {1: 15.0, 6: 35.0, 24: 55.0}
                base_margin = margin_map.get(horizon, 45.0)
                z_factor = 1.645 if nominal_coverage <= 0.90 else 1.96
                margin = base_margin * (z_factor / 1.645)

                lower_bounds = [round(max(0.0, p - margin), 2) for p in point_preds]
                upper_bounds = [round(p + margin, 2) for p in point_preds]
                interval_method = "heuristic_fallback"
                is_calibrated = False

        tot_lat = round((time.perf_counter() - t0) * 1000.0, 4)
        per_samp = round(tot_lat / n_samples, 4) if n_samples > 0 else 0.0

        return point_preds, lower_bounds, upper_bounds, tot_lat, per_samp, interval_method, is_calibrated
