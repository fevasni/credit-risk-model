"""
Test script for the data processing pipeline.

This script validates that the preprocessing pipeline works correctly
and produces model-ready features from raw transaction data.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_raw_data
from src.data_processing import (
    fit_preprocessing_pipeline,
    process_data_to_model_ready,
    get_feature_importance_woe,
    get_numerical_features,
    get_categorical_features
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run tests on the preprocessing pipeline."""
    
    print("\n" + "=" * 80)
    print("DATA PROCESSING PIPELINE - VALIDATION TEST")
    print("=" * 80 + "\n")
    
    try:
        # =====================================================================
        # Step 1: Load raw data
        # =====================================================================
        logger.info("Step 1: Loading raw data...")
        df_raw = load_raw_data(validate=True)
        
        logger.info(f"✓ Loaded {len(df_raw):,} transactions")
        logger.info(f"  Shape: {df_raw.shape}")
        logger.info(f"  Columns: {list(df_raw.columns)}")
        logger.info(f"  Memory usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # =====================================================================
        # Step 2: Create and fit preprocessing pipeline
        # =====================================================================
        logger.info("\nStep 2: Creating and fitting preprocessing pipeline...")
        
        # Use only a sample for faster testing (optional)
        df_sample = df_raw.sample(min(1000, len(df_raw)), random_state=42)
        
        pipeline = fit_preprocessing_pipeline(
            df_sample,
            target_col='FraudResult',
            include_woe=True,
            include_log_transform=True
        )
        
        logger.info(f"✓ Pipeline fitted successfully")
        logger.info(f"  Pipeline steps: {list(pipeline.named_steps.keys())}")
        
        # =====================================================================
        # Step 3: Transform data
        # =====================================================================
        logger.info("\nStep 3: Transforming data with fitted pipeline...")
        
        df_processed, _ = process_data_to_model_ready(
            df_sample,
            pipeline=pipeline,
            fit_on_data=False
        )
        
        logger.info(f"✓ Data processed successfully")
        logger.info(f"  Original shape: {df_sample.shape}")
        logger.info(f"  Processed shape: {df_processed.shape}")
        logger.info(f"  Features generated: {df_processed.shape[1]} columns")
        
        # =====================================================================
        # Step 4: Analyze processed data
        # =====================================================================
        logger.info("\nStep 4: Analyzing processed data...")
        
        # Data types
        numeric_features = get_numerical_features(df_processed)
        categorical_features = get_categorical_features(df_processed)
        
        logger.info(f"  Numerical features: {len(numeric_features)}")
        logger.info(f"  Categorical features: {len(categorical_features)}")
        
        # Missing values
        missing_pct = (df_processed.isna().sum() / len(df_processed) * 100)
        if missing_pct.sum() > 0:
            logger.warning(f"  Missing values detected:")
            for col, pct in missing_pct[missing_pct > 0].items():
                logger.warning(f"    {col}: {pct:.2f}%")
        else:
            logger.info(f"  ✓ No missing values")
        
        # Summary statistics
        logger.info(f"\n  Numerical features summary:")
        logger.info(df_processed[numeric_features].describe().to_string())
        
        # =====================================================================
        # Step 5: Extract feature importance
        # =====================================================================
        logger.info("\nStep 5: Extracting WoE/IV feature importance...")
        
        importance_df = get_feature_importance_woe(pipeline)
        
        if not importance_df.empty:
            logger.info(f"✓ Extracted importance for {len(importance_df)} features")
            logger.info("\n  Top 10 features by Information Value (IV):")
            logger.info(importance_df.head(10).to_string(index=False))
        else:
            logger.warning("  No WoE/IV importance available")
        
        # =====================================================================
        # Step 6: Sample output
        # =====================================================================
        logger.info("\nStep 6: Sample of processed data:")
        logger.info(f"\n{df_processed.head(10).to_string()}")
        
        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"[OK] Raw data loaded: {len(df_raw):,} transactions")
        print(f"[OK] Pipeline created with {len(pipeline.steps)} transformation steps")
        print(f"[OK] Data processed: {df_processed.shape[0]:,} records x {df_processed.shape[1]} features")
        print(f"[OK] Numerical features: {len(numeric_features)}")
        print(f"[OK] Categorical features: {len(categorical_features)}")
        print(f"[OK] Missing values: {missing_pct.sum():.2f}%")
        print("\n[SUCCESS] PIPELINE VALIDATION SUCCESSFUL!")
        print("=" * 80 + "\n")
        
        return True
    
    except Exception as e:
        print("\n" + "=" * 80)
        print("[FAILED] VALIDATION FAILED")
        print("=" * 80)
        logger.error(f"Error during validation: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
