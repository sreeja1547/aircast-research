"""Abstract base class for all AirCast forecasting models.

Provides standardized interface for training, prediction, persistence, validation,
and model metadata tracking across baseline, tree-based, and deep learning architectures.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract Base Class for multi-horizon air quality forecasters.

    Enforces uniform API: fit, predict, save, load, and diagnostic introspection.

    Attributes
    ----------
    name : str
        Human-readable identifier for the model (e.g., 'xgboost_h24').
    model_type : str
        Taxonomic category: 'baseline', 'linear', 'tree', 'deep_learning'.
    feature_names : List[str]
        Ordered list of feature column names seen during fit.
    is_fitted : bool
        Flag indicating if model parameters have been estimated.
    metadata : Dict[str, Any]
        Arbitrary training metadata (hyperparameters, training time, metrics).
    """

    def __init__(self, name: str, model_type: str = "generic") -> None:
        self.name = name
        self.model_type = model_type
        self.feature_names: List[str] = []
        self.categorical_encoders_: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False
        self.metadata: Dict[str, Any] = {}

    def _validate_inputs(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        is_training: bool = False,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Validate input dimensions, types, and schema consistency.

        Converts categorical / string columns (e.g. station identifiers) into
        deterministic numeric encodings, and guarantees column order consistency.
        """
        if X is None or len(X) == 0:
            raise ValueError(f"[{self.name}] Input feature matrix X cannot be empty.")

        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            if is_training:
                self.feature_names = list(X_out.columns)
                self.categorical_encoders_ = {}

                # Detect and encode non-numeric columns
                for col in X_out.columns:
                    if not pd.api.types.is_numeric_dtype(X_out[col].dtype):
                        unique_vals = sorted([str(v) for v in X_out[col].dropna().unique()])
                        mapping = {v: float(i) for i, v in enumerate(unique_vals)}
                        self.categorical_encoders_[col] = mapping
                        X_out[col] = X_out[col].astype(str).map(mapping).fillna(-1.0).astype(np.float64)

                if y is not None:
                    if len(X_out) != len(y):
                        raise ValueError(
                            f"[{self.name}] Length mismatch: X has {len(X_out)} rows, y has {len(y)} rows."
                        )
                    y_arr = np.asarray(y)
                    if np.isnan(y_arr).any():
                        raise ValueError(
                            f"[{self.name}] Target y contains NaNs. Filter missing targets prior to fitting."
                        )
            else:
                if not self.is_fitted:
                    raise RuntimeError(
                        f"[{self.name}] Cannot predict with an unfitted model. Call fit() first."
                    )

                missing_cols = set(self.feature_names) - set(X_out.columns)
                if missing_cols:
                    raise ValueError(
                        f"[{self.name}] Prediction data missing required features: {missing_cols}"
                    )

                # Maintain exact training column ordering
                X_out = X_out[self.feature_names].copy()

                # Apply fitted categorical encodings (unseen categories mapped to -1.0)
                for col, mapping in self.categorical_encoders_.items():
                    if col in X_out.columns:
                        X_out[col] = X_out[col].astype(str).map(mapping).fillna(-1.0).astype(np.float64)

            return X_out

        else:
            # Handle NumPy array
            X_arr = np.asarray(X)
            if is_training:
                self.feature_names = [f"f_{i}" for i in range(X_arr.shape[1])]
                if y is not None:
                    if len(X_arr) != len(y):
                        raise ValueError(
                            f"[{self.name}] Length mismatch: X has {len(X_arr)} rows, y has {len(y)} rows."
                        )
                    y_arr = np.asarray(y)
                    if np.isnan(y_arr).any():
                        raise ValueError(
                            f"[{self.name}] Target y contains NaNs."
                        )
            else:
                if not self.is_fitted:
                    raise RuntimeError(
                        f"[{self.name}] Cannot predict with an unfitted model. Call fit() first."
                    )
            return X_arr

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "BaseForecaster":
        """Fit model parameters to historical training observations.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training feature matrix.
        y : pd.Series or np.ndarray
            Training target PM2.5 vector.
        X_val : pd.DataFrame or np.ndarray, optional
            Validation feature matrix for early stopping / tuning.
        y_val : pd.Series or np.ndarray, optional
            Validation target vector.

        Returns
        -------
        self : BaseForecaster
        """
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate point predictions for given feature inputs.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            1D array of predicted PM2.5 concentrations in ug/m^3.
        """
        pass

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model artifact and metadata to disk.

        Parameters
        ----------
        filepath : str or Path
            Target destination file path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseForecaster":
        """Deserialize a saved forecaster model instance from disk.

        Parameters
        ----------
        filepath : str or Path
            File path of saved model artifact.

        Returns
        -------
        BaseForecaster
            Restored model instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        model = joblib.load(path)
        if not isinstance(model, BaseForecaster):
            raise TypeError(
                f"Loaded object of type {type(model)} does not inherit from BaseForecaster."
            )
        return model

    def get_model_size_mb(self, filepath: Optional[Union[str, Path]] = None) -> float:
        """Calculate model artifact size on disk or estimate in memory.

        Parameters
        ----------
        filepath : str or Path, optional
            File path if already serialized.

        Returns
        -------
        float
            Size in megabytes (MB).
        """
        if filepath is not None and os.path.exists(filepath):
            return round(os.path.getsize(filepath) / (1024.0 * 1024.0), 4)

        # Estimate via temporary buffer
        import io
        buf = io.BytesIO()
        joblib.dump(self, buf)
        return round(len(buf.getvalue()) / (1024.0 * 1024.0), 4)
