"""Prepare and cache feature-engineered multi-horizon dataset.

Guarantees:
- StationAwareCausalImputer is fitted strictly on the Training partition (<= 2015-12-31 23:00:00).
- Targets target_h1, target_h6, target_h24 are created from raw un-imputed PM2.5.
- Caches output into data/processed/forecasting_features_cache.parquet for fast experiment iteration.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.data.loader import load_all_stations
from src.data.forecasting_dataset import MultiHorizonForecastingDataset
from src.preprocessing.cleaner import StationAwareCausalImputer
from src.preprocessing.splitter import TemporalPurgedSplitter


def main():
    print("=" * 70)
    print("AirCast: Feature Engineering & Dataset Caching Pipeline")
    print("=" * 70)

    t0 = time.perf_counter()
    raw_data_dir = "PRSA_Data_20130301-20170228"
    cache_path = Path("data/processed/forecasting_features_cache.parquet")

    # Step 1: Load raw stations
    print(f"\n[1/4] Loading raw station records from '{raw_data_dir}'...")
    raw_df = load_all_stations(raw_data_dir)
    print(f"      Loaded {len(raw_df):,} rows across {raw_df['station'].nunique()} stations.")

    # Step 2: Establish strict training partition for imputer fitting
    splitter = TemporalPurgedSplitter()
    dt_series = pd.to_datetime(raw_df["datetime"])
    train_raw = raw_df[dt_series <= splitter.train_end].copy()
    print(f"\n[2/4] Fitting StationAwareCausalImputer on Train partition ({len(train_raw):,} rows)...")
    imputer = StationAwareCausalImputer()
    imputer.fit(train_raw)
    print("      Imputer successfully fitted with training-only statistics.")

    # Step 3: Run MultiHorizonForecastingDataset pipeline
    print("\n[3/4] Building multi-horizon targets and full 47 feature columns...")
    dataset_builder = MultiHorizonForecastingDataset(horizons=[1, 6, 24])
    prepared_df = dataset_builder.prepare_dataset(
        raw_df=raw_df,
        imputer=imputer,
        fit_df=train_raw,
    )
    print(f"      Constructed prepared DataFrame with shape: {prepared_df.shape}")

    # Step 4: Save to Parquet
    print(f"\n[4/4] Writing cached artifact to '{cache_path}'...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_parquet(cache_path, engine="pyarrow", index=False)
    file_size_mb = cache_path.stat().st_size / (1024 * 1024)

    elapsed = time.perf_counter() - t0
    print(f"\n[SUCCESS] Feature cache generated in {elapsed:.2f}s! Size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
