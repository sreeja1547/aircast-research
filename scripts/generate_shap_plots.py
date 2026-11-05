"""Generate publication TreeSHAP feature attributions and dependence plots.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.explainability.shap_engine import ShapExplanationEngine
from src.models import XGBoostForecaster
from src.preprocessing.splitter import TemporalPurgedSplitter


def main():
    print("Generating TreeSHAP attributions and summary plot...")
    cache_path = "data/processed/forecasting_features_cache.parquet"
    df_prepared = pd.read_parquet(cache_path)

    dataset_builder = MultiHorizonForecastingDataset()
    splitter = TemporalPurgedSplitter()

    X, y, metadata = dataset_builder.get_horizon_dataset(
        df_prepared, horizon=24, feature_group="combined_all"
    )

    dt_series = pd.to_datetime(metadata[splitter.datetime_col])
    test_mask = (dt_series >= splitter.test_start) & (dt_series <= splitter.test_end)
    X_test = X[test_mask]

    # Load trained model
    model_path = "artifacts/models/xgboost_h24.joblib"
    model = XGBoostForecaster.load(model_path)

    engine = ShapExplanationEngine(model)
    # Explain a representative sample of 1,000 test points
    sample_eval = X_test.iloc[:1000]
    engine.generate_summary_plot(
        sample_eval,
        output_path="artifacts/plots/shap_summary.png",
        top_k=15,
    )
    print("[SUCCESS] SHAP summary plot generated at 'artifacts/plots/shap_summary.png'!")


if __name__ == "__main__":
    main()
