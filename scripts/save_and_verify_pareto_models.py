"""Train and serialize the recommended Pareto models (xgboost_pruned_h6 and xgboost_fast_h24)
without overwriting or deleting any existing baseline models.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import joblib
import numpy as np
import pandas as pd

from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.preprocessing.splitter import TemporalPurgedSplitter
from src.features.registry import get_feature_group
from src.models.trees import XGBoostForecaster
from src.evaluation.metrics import calculate_regression_metrics

def main():
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    cache_path = "data/processed/forecasting_features_cache.parquet"
    print(f"Loading cached features from '{cache_path}'...")
    df_prepared = pd.read_parquet(cache_path)

    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    # 1. Horizon 6: XGBoost_Pruned (90 trees, depth 6, lr 0.08)
    pruned_path = models_dir / "xgboost_pruned_h6.joblib"
    print(f"\n[1/2] Processing {pruned_path.name}...")
    X_full6, y_full6, meta6 = dataset_builder.get_horizon_dataset(
        df_prepared, horizon=6, feature_group="combined_all"
    )
    for col in X_full6.columns:
        if not pd.api.types.is_numeric_dtype(X_full6[col].dtype):
            uvals = sorted([str(v) for v in X_full6[col].dropna().unique()])
            smap = {v: float(i) for i, v in enumerate(uvals)}
            X_full6[col] = X_full6[col].astype(str).map(smap).fillna(-1.0).astype(np.float32)

    dt_series6 = pd.to_datetime(meta6[splitter.datetime_col])
    train_mask6 = dt_series6 <= splitter.train_end
    val_mask6 = (dt_series6 >= splitter.val_start) & (dt_series6 <= splitter.val_end)
    test_mask6 = (dt_series6 >= splitter.test_start) & (dt_series6 <= splitter.test_end)

    X_train6, y_train6 = X_full6[train_mask6], y_full6[train_mask6]
    X_val6, y_val6 = X_full6[val_mask6], y_full6[val_mask6]
    X_test6, y_test6 = X_full6[test_mask6], y_full6[test_mask6]

    m_pruned = XGBoostForecaster(
        name="xgboost_pruned_h6",
        n_estimators=90,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=12,
        tree_method="hist",
        random_state=42,
    )
    m_pruned.fit(X_train6, y_train6, X_val=X_val6, y_val=y_val6)
    preds6 = m_pruned.predict(X_test6)
    metrics6 = calculate_regression_metrics(y_test6, preds6)
    print(f"  h=6 XGBoost_Pruned Test MAE: {metrics6['mae']:.4f}, RMSE: {metrics6['rmse']:.4f}, R2: {metrics6['r2']:.4f}")

    joblib.dump(m_pruned, pruned_path)
    file_size_mb = pruned_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {pruned_path} (Size: {file_size_mb:.4f} MB)")

    # 2. Horizon 24: XGBoost_Fast (50 trees, depth 5, lr 0.12)
    fast_path = models_dir / "xgboost_fast_h24.joblib"
    print(f"\n[2/2] Processing {fast_path.name}...")
    X_full24, y_full24, meta24 = dataset_builder.get_horizon_dataset(
        df_prepared, horizon=24, feature_group="combined_all"
    )
    for col in X_full24.columns:
        if not pd.api.types.is_numeric_dtype(X_full24[col].dtype):
            uvals = sorted([str(v) for v in X_full24[col].dropna().unique()])
            smap = {v: float(i) for i, v in enumerate(uvals)}
            X_full24[col] = X_full24[col].astype(str).map(smap).fillna(-1.0).astype(np.float32)

    dt_series24 = pd.to_datetime(meta24[splitter.datetime_col])
    train_mask24 = dt_series24 <= splitter.train_end
    val_mask24 = (dt_series24 >= splitter.val_start) & (dt_series24 <= splitter.val_end)
    test_mask24 = (dt_series24 >= splitter.test_start) & (dt_series24 <= splitter.test_end)

    X_train24, y_train24 = X_full24[train_mask24], y_full24[train_mask24]
    X_val24, y_val24 = X_full24[val_mask24], y_full24[val_mask24]
    X_test24, y_test24 = X_full24[test_mask24], y_full24[test_mask24]

    m_fast = XGBoostForecaster(
        name="xgboost_fast_h24",
        n_estimators=50,
        max_depth=5,
        learning_rate=0.12,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
        tree_method="hist",
        random_state=42,
    )
    m_fast.fit(X_train24, y_train24, X_val=X_val24, y_val=y_val24)
    preds24 = m_fast.predict(X_test24)
    metrics24 = calculate_regression_metrics(y_test24, preds24)
    print(f"  h=24 XGBoost_Fast Test MAE: {metrics24['mae']:.4f}, RMSE: {metrics24['rmse']:.4f}, R2: {metrics24['r2']:.4f}")

    joblib.dump(m_fast, fast_path)
    file_size_fast_mb = fast_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {fast_path} (Size: {file_size_fast_mb:.4f} MB)")

    print("\nVerification of all 3 Pareto models on disk:")
    h1_path = models_dir / "xgboost_h1.joblib"
    print(f"  - h1  (Baseline): {h1_path.name} -> {h1_path.stat().st_size / (1024*1024):.4f} MB")
    print(f"  - h6  (Pruned)  : {pruned_path.name} -> {file_size_mb:.4f} MB")
    print(f"  - h24 (Fast)    : {fast_path.name} -> {file_size_fast_mb:.4f} MB")
    total_pareto_mb = (h1_path.stat().st_size + pruned_path.stat().st_size + fast_path.stat().st_size) / (1024 * 1024)
    print(f"  Total Recommended Pareto Suite: {total_pareto_mb:.4f} MB")

if __name__ == "__main__":
    main()
