"""PyTorch LSTM Forecaster for multi-horizon air quality forecasting.

Implements a recurrent neural network architecture with sequence windowing,
dropout regularization, AdamW optimization, gradient clipping, and validation
early stopping without crossing station or temporal partition boundaries.
"""

import copy
import io
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.base import BaseForecaster


class PyTorchLSTMModel(nn.Module):
    """PyTorch Recurrent LSTM Neural Network.

    Processes sequential feature representations (batch_size, seq_len, n_features),
    extracting temporal dependencies through stacked LSTM cells followed by
    non-linear projection layers.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int, default=64
        Hidden state dimensionality of the LSTM cells.
    num_layers : int, default=2
        Number of stacked LSTM layers.
    dropout : float, default=0.2
        Dropout probability between LSTM layers and dense head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, input_dim)

        Returns
        -------
        torch.Tensor of shape (batch_size, 1)
        """
        out, (h_n, _) = self.lstm(x)
        # Use last hidden state of top LSTM layer: (batch_size, hidden_dim)
        last_hidden = h_n[-1]
        out = self.head(last_hidden)
        return out


class TabularSequenceDataset(Dataset):
    """Dataset converting tabular or sequence arrays into PyTorch Tensors."""

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        seq_len: int = 1,
    ) -> None:
        if X.ndim == 2:
            # Reshape tabular (N, F) into (N, 1, F)
            self.X = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)
        elif X.ndim == 3:
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            raise ValueError(f"Expected 2D or 3D array for X, got ndim={X.ndim}")

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class LSTMForecaster(BaseForecaster):
    """Deep Learning LSTM Forecaster with early stopping and AdamW optimization.

    Supports both tabular lag feature matrices and 3D sequence tensors. When
    provided with tabular features, treats the feature vector as an instant sequence
    or past lag trajectory.

    Parameters
    ----------
    hidden_dim : int, default=64
        Number of hidden units in each LSTM layer.
    num_layers : int, default=2
        Number of recurrent LSTM layers.
    dropout : float, default=0.2
        Dropout rate for regularizing recurrent and dense layers.
    learning_rate : float, default=1e-3
        Initial learning rate for AdamW optimizer.
    weight_decay : float, default=1e-4
        L2 weight regularization penalty.
    batch_size : int, default=256
        Batch size for gradient updates.
    max_epochs : int, default=30
        Maximum training epochs.
    patience : int, default=5
        Number of epochs with no improvement before early stopping.
    gradient_clip_val : float, default=1.0
        Maximum norm for gradient clipping.
    random_state : int, default=42
        Seed for pseudo-random number generator.
    device : Optional[str], default=None
        Execution device: 'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        name: str = "lstm",
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 30,
        patience: int = 5,
        gradient_clip_val: float = 1.0,
        random_state: int = 42,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, model_type="deep_learning")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip_val = gradient_clip_val
        self.random_state = random_state

        if device is None:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device

        self.torch_device = torch.device(self.device_str)
        self.network: Optional[PyTorchLSTMModel] = None
        self.scaler_mean_: Optional[np.ndarray] = None
        self.scaler_scale_: Optional[np.ndarray] = None

    def _set_seeds(self) -> None:
        """Enforce deterministic behavior across PyTorch and NumPy."""
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit feature standardization parameters."""
        if X.ndim == 2:
            self.scaler_mean_ = np.mean(X, axis=0)
            self.scaler_scale_ = np.std(X, axis=0)
            self.scaler_scale_[self.scaler_scale_ < 1e-6] = 1.0
            return (X - self.scaler_mean_) / self.scaler_scale_
        elif X.ndim == 3:
            # Standardize across (samples, time)
            self.scaler_mean_ = np.mean(X, axis=(0, 1))
            self.scaler_scale_ = np.std(X, axis=(0, 1))
            self.scaler_scale_[self.scaler_scale_ < 1e-6] = 1.0
            return (X - self.scaler_mean_) / self.scaler_scale_
        return X

    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted standardization."""
        if self.scaler_mean_ is None or self.scaler_scale_ is None:
            return X
        return (X - self.scaler_mean_) / self.scaler_scale_

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "LSTMForecaster":
        """Fit LSTM model parameters using AdamW with early stopping."""
        t0 = time.perf_counter()
        self._set_seeds()

        X_valid = self._validate_inputs(X, y, is_training=True)
        y_arr = np.asarray(y, dtype=np.float32)

        if isinstance(X_valid, pd.DataFrame):
            X_arr = X_valid.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X_valid, dtype=np.float32)

        X_scaled = self._standardize_fit(X_arr)
        input_dim = X_scaled.shape[-1]

        # Initialize PyTorch network
        self.network = PyTorchLSTMModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.torch_device)

        train_dataset = TabularSequenceDataset(X_scaled, y_arr)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            self.is_fitted = True
            X_v_df = self._validate_inputs(X_val, is_training=False)
            self.is_fitted = False
            if isinstance(X_v_df, pd.DataFrame):
                X_v = X_v_df.to_numpy(dtype=np.float32)
            else:
                X_v = np.asarray(X_v_df, dtype=np.float32)
            y_v = np.asarray(y_val, dtype=np.float32)
            X_v_scaled = self._standardize_transform(X_v)
            val_dataset = TabularSequenceDataset(X_v_scaled, y_v)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state_dict = None
        patience_counter = 0
        history: List[Dict[str, float]] = []

        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.torch_device)
                batch_y = batch_y.to(self.torch_device)

                optimizer.zero_grad()
                preds = self.network(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()

                if self.gradient_clip_val > 0.0:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), max_norm=self.gradient_clip_val
                    )

                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            # Validation evaluation
            if val_loader is not None:
                self.network.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.torch_device)
                        batch_y = batch_y.to(self.torch_device)
                        val_preds = self.network(batch_x)
                        val_loss = criterion(val_preds, batch_y)
                        val_losses.append(val_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                val_rmse = float(np.sqrt(avg_val_loss))
                scheduler.step(avg_val_loss)

                history.append(
                    {
                        "epoch": epoch,
                        "train_mse": round(avg_train_loss, 4),
                        "val_mse": round(avg_val_loss, 4),
                        "val_rmse": round(val_rmse, 4),
                    }
                )

                if avg_val_loss < best_val_loss - 1e-4:
                    best_val_loss = avg_val_loss
                    best_state_dict = copy.deepcopy(self.network.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
            else:
                history.append({"epoch": epoch, "train_mse": round(avg_train_loss, 4)})

        if best_state_dict is not None and self.network is not None:
            self.network.load_state_dict(best_state_dict)

        self.is_fitted = True
        self.metadata = {
            "training_time_sec": round(time.perf_counter() - t0, 4),
            "n_samples": len(X_arr),
            "n_features": len(self.feature_names),
            "device": self.device_str,
            "final_epoch": len(history),
            "best_val_loss": round(best_val_loss, 4) if val_loader is not None else None,
            "best_val_rmse": round(float(np.sqrt(best_val_loss)), 4)
            if val_loader is not None
            else None,
            "history": history,
        }
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate PM2.5 point predictions."""
        X_valid = self._validate_inputs(X, is_training=False)

        if isinstance(X_valid, pd.DataFrame):
            X_arr = X_valid.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X_valid, dtype=np.float32)

        X_scaled = self._standardize_transform(X_arr)
        dataset = TabularSequenceDataset(X_scaled)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.network is None:
            raise RuntimeError("Model network is not initialized.")

        self.network.eval()
        predictions: List[np.ndarray] = []

        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.torch_device)
                preds = self.network(batch_x)
                predictions.append(preds.cpu().numpy().ravel())

        all_preds = np.concatenate(predictions)
        return np.maximum(0.0, all_preds)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize PyTorch LSTM model and metadata."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "name": self.name,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "categorical_encoders": getattr(self, "categorical_encoders_", {}),
            "is_fitted": self.is_fitted,
            "metadata": self.metadata,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "device_str": self.device_str,
            "scaler_mean": self.scaler_mean_,
            "scaler_scale": self.scaler_scale_,
            "network_state_dict": self.network.state_dict() if self.network else None,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LSTMForecaster":
        """Load serialized PyTorch LSTM forecaster."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        state = joblib.load(path)
        forecaster = cls(
            name=state["name"],
            hidden_dim=state["hidden_dim"],
            num_layers=state["num_layers"],
            dropout=state["dropout"],
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            batch_size=state["batch_size"],
            random_state=state["random_state"],
            device=state.get("device_str", "cpu"),
        )
        forecaster.feature_names = state["feature_names"]
        forecaster.categorical_encoders_ = state.get("categorical_encoders", {})
        forecaster.is_fitted = state["is_fitted"]
        forecaster.metadata = state["metadata"]
        forecaster.scaler_mean_ = state["scaler_mean"]
        forecaster.scaler_scale_ = state["scaler_scale"]

        if state["network_state_dict"] is not None:
            input_dim = len(forecaster.feature_names)
            forecaster.network = PyTorchLSTMModel(
                input_dim=input_dim,
                hidden_dim=forecaster.hidden_dim,
                num_layers=forecaster.num_layers,
                dropout=forecaster.dropout,
            ).to(forecaster.torch_device)
            forecaster.network.load_state_dict(state["network_state_dict"])
            forecaster.network.eval()

        return forecaster
