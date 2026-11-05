"""Comprehensive evaluation metrics and latency benchmarking for AirCast Research.

Implements mathematically rigorous regression metrics, extreme pollution episode
partitioning, subgroup evaluations, and latency/throughput profiling.
"""

import time
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    extreme_threshold: float = 150.0,
) -> Dict[str, float]:
    """Compute comprehensive regression metrics and extreme-episode statistics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth PM2.5 concentrations in ug/m^3.
    y_pred : array-like of shape (n_samples,)
        Predicted PM2.5 concentrations in ug/m^3.
    extreme_threshold : float, default=150.0
        Threshold in ug/m^3 defining severe pollution episodes (e.g., AQI > 200).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - r2: Coefficient of determination
        - mape: Mean Absolute Percentage Error (with epsilon floor)
        - smape: Symmetric Mean Absolute Percentage Error (0-100%)
        - pearson_r: Pearson correlation coefficient
        - sample_count: Total evaluated sample count
        - extreme_count: Count of extreme ground-truth samples
        - extreme_mae: MAE on extreme samples (NaN if extreme_count == 0)
        - extreme_rmse: RMSE on extreme samples (NaN if extreme_count == 0)
        - normal_count: Count of normal ground-truth samples
        - normal_mae: MAE on normal samples (NaN if normal_count == 0)
        - normal_rmse: RMSE on normal samples (NaN if normal_count == 0)
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_t)} samples, y_pred has {len(y_p)} samples."
        )

    if len(y_t) == 0:
        raise ValueError("Cannot calculate metrics on empty arrays.")

    # Mask valid finite pairs
    valid_mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.all(valid_mask):
        y_t = y_t[valid_mask]
        y_p = y_p[valid_mask]

    n_samples = len(y_t)
    if n_samples == 0:
        raise ValueError("No valid finite samples to evaluate.")

    # Core regression metrics
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # R2 score handling edge case of zero variance
    if np.var(y_t) < 1e-12:
        r2 = 0.0 if np.allclose(y_t, y_p) else -1.0
    else:
        r2 = float(r2_score(y_t, y_p))

    # MAPE (with small epsilon floor to prevent division by zero)
    eps = 1e-5
    abs_errors = np.abs(y_t - y_p)
    mape = float(np.mean(abs_errors / np.maximum(np.abs(y_t), eps)) * 100.0)

    # Symmetric MAPE (sMAPE): 100/n * sum( 2 * |y - y_hat| / (|y| + |y_hat| + eps) )
    denominator = (np.abs(y_t) + np.abs(y_p)) / 2.0 + eps
    smape = float(np.mean(abs_errors / denominator) * 100.0)

    # Pearson correlation coefficient
    if np.var(y_t) < 1e-12 or np.var(y_p) < 1e-12:
        pearson_r = 0.0
    else:
        r, _ = stats.pearsonr(y_t, y_p)
        pearson_r = float(r) if np.isfinite(r) else 0.0

    # Subgroup analysis: Extreme vs Normal pollution episodes
    extreme_mask = y_t >= extreme_threshold
    normal_mask = ~extreme_mask

    extreme_count = int(np.sum(extreme_mask))
    normal_count = int(np.sum(normal_mask))

    if extreme_count > 0:
        extreme_mae = float(mean_absolute_error(y_t[extreme_mask], y_p[extreme_mask]))
        extreme_rmse = float(np.sqrt(mean_squared_error(y_t[extreme_mask], y_p[extreme_mask])))
    else:
        extreme_mae = float("nan")
        extreme_rmse = float("nan")

    if normal_count > 0:
        normal_mae = float(mean_absolute_error(y_t[normal_mask], y_p[normal_mask]))
        normal_rmse = float(np.sqrt(mean_squared_error(y_t[normal_mask], y_p[normal_mask])))
    else:
        normal_mae = float("nan")
        normal_rmse = float("nan")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
        "smape": round(smape, 4),
        "pearson_r": round(pearson_r, 4),
        "sample_count": n_samples,
        "extreme_count": extreme_count,
        "extreme_mae": round(extreme_mae, 4) if np.isfinite(extreme_mae) else None,
        "extreme_rmse": round(extreme_rmse, 4) if np.isfinite(extreme_rmse) else None,
        "normal_count": normal_count,
        "normal_mae": round(normal_mae, 4) if np.isfinite(normal_mae) else None,
        "normal_rmse": round(normal_rmse, 4) if np.isfinite(normal_rmse) else None,
    }


def evaluate_by_subgroups(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    groups: Union[np.ndarray, pd.Series],
) -> pd.DataFrame:
    """Evaluate regression metrics grouped by a categorical variable (e.g., station).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    groups : array-like of shape (n_samples,)
        Group indicator (e.g., station name or month).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by group containing metrics for each group.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    grp = np.asarray(groups).ravel()

    unique_groups = np.unique(grp)
    records = []

    for g in unique_groups:
        mask = grp == g
        if np.sum(mask) == 0:
            continue
        metrics = calculate_regression_metrics(y_t[mask], y_p[mask])
        metrics["group"] = g
        records.append(metrics)

    res_df = pd.DataFrame(records)
    if not res_df.empty and "group" in res_df.columns:
        res_df = res_df.set_index("group")
    return res_df


def benchmark_latency(
    model: Any,
    X_sample: Union[np.ndarray, pd.DataFrame],
    n_iterations: int = 50,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """Measure inference latency and throughput under controlled iterations.

    Parameters
    ----------
    model : object
        Forecaster model with a .predict() method.
    X_sample : array-like of shape (n_samples, n_features)
        Input features for latency profiling.
    n_iterations : int, default=50
        Number of warm-up and timing iterations.
    batch_size : Optional[int], default=None
        If provided, slices X_sample into batches of this size.

    Returns
    -------
    Dict[str, float]
        Dictionary of latency statistics (in ms) and throughput (samples/sec).
    """
    if isinstance(X_sample, pd.DataFrame):
        X_mat = X_sample.copy()
    else:
        X_mat = np.asarray(X_sample)

    n_samples = len(X_mat)
    if n_samples == 0:
        raise ValueError("Cannot benchmark on empty sample.")

    eval_data = X_mat if batch_size is None else X_mat[:batch_size]
    eval_count = len(eval_data)

    # Warm-up run (3 iterations)
    for _ in range(3):
        _ = model.predict(eval_data)

    latencies_sec = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = model.predict(eval_data)
        t1 = time.perf_counter()
        latencies_sec.append(t1 - t0)

    latencies_ms = np.array(latencies_sec) * 1000.0

    mean_batch_ms = float(np.mean(latencies_ms))
    p50_batch_ms = float(np.median(latencies_ms))
    p95_batch_ms = float(np.percentile(latencies_ms, 95))
    p99_batch_ms = float(np.percentile(latencies_ms, 99))
    min_batch_ms = float(np.min(latencies_ms))
    max_batch_ms = float(np.max(latencies_ms))

    # Per-sample latency and throughput
    mean_per_sample_ms = mean_batch_ms / eval_count
    throughput_samples_per_sec = float(eval_count / (mean_batch_ms / 1000.0))

    return {
        "batch_sample_count": eval_count,
        "n_iterations": n_iterations,
        "mean_batch_latency_ms": round(mean_batch_ms, 4),
        "p50_batch_latency_ms": round(p50_batch_ms, 4),
        "p95_batch_latency_ms": round(p95_batch_ms, 4),
        "p99_batch_latency_ms": round(p99_batch_ms, 4),
        "min_batch_latency_ms": round(min_batch_ms, 4),
        "max_batch_latency_ms": round(max_batch_ms, 4),
        "mean_per_sample_latency_ms": round(mean_per_sample_ms, 6),
        "throughput_samples_per_sec": round(throughput_samples_per_sec, 2),
    }
