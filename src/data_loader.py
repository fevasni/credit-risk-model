"""
Data loading and preprocessing module for credit risk modeling.

This module handles reading data from files, applying initial transformations,
and ensuring data integrity through defensive checks and validation.

Key responsibilities:
- Load data from CSV files with error handling
- Apply type conversions (DateTime, categorical)
- Execute defensive validation checks
- Provide data quality summaries
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from src.constants import RAW_DATA_FILE, DATA_DICTIONARY_FILE
from src.validators import (
    validate_dataframe_structure, detect_missing_values,
    detect_duplicates, run_full_validation, ValidationError
)

# Setup logging
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Optional[Path] = None, 
                  validate: bool = True) -> pd.DataFrame:
    """
    Load raw transaction data from CSV file with defensive checks.

    Args:
        filepath: Path to CSV file (default: RAW_DATA_FILE from constants)
        validate: Whether to run full validation pipeline (default: True)

    Returns:
        Loaded DataFrame with applied transformations

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be read as CSV
        ValidationError: If validation fails (when validate=True)

    Example:
        >>> df = load_raw_data()  # Load from default path
        >>> df = load_raw_data(filepath=Path('data/custom.csv'), validate=False)
    """
    filepath = filepath or RAW_DATA_FILE
    filepath = Path(filepath)

    # Defensive check: File existence
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found at {filepath}\n"
            f"Expected location: {RAW_DATA_FILE}"
        )

    logger.info(f"Loading data from: {filepath}")

    try:
        # Load CSV with defensive defaults
        df = pd.read_csv(
            filepath,
            dtype_backend='numpy_nullable',  # Handle nulls gracefully
            low_memory=False  # Prevent mixed dtype inference warnings
        )
    except Exception as e:
        raise ValueError(f"Failed to read CSV from {filepath}: {str(e)}")

    logger.info(f"✓ Raw data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Apply basic structure validation
    try:
        validate_dataframe_structure(df)
    except ValidationError as e:
        logger.error(f"Structure validation failed: {str(e)}")
        raise

    # Apply initial transformations
    df = _apply_type_conversions(df)

    # Run full validation pipeline if requested
    if validate:
        logger.info("Running full validation pipeline...")
        validation_results = run_full_validation(df)
        logger.info(f"Validation results: {validation_results}")

    return df


def load_data_dictionary(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load data dictionary/schema documentation.

    Args:
        filepath: Path to data dictionary CSV (default: DATA_DICTIONARY_FILE)

    Returns:
        DataFrame containing variable definitions

    Raises:
        FileNotFoundError: If dictionary file not found
        ValueError: If file cannot be read

    Example:
        >>> data_dict = load_data_dictionary()
    """
    filepath = filepath or DATA_DICTIONARY_FILE
    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning(f"Data dictionary not found at {filepath}")
        return pd.DataFrame()

    try:
        data_dict = pd.read_csv(filepath)
        logger.info(f"✓ Data dictionary loaded: {len(data_dict)} variable definitions")
        return data_dict
    except Exception as e:
        logger.error(f"Failed to load data dictionary: {str(e)}")
        raise ValueError(f"Cannot read data dictionary from {filepath}: {str(e)}")


def _apply_type_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply type conversions and formatting to DataFrame columns.

    Defensive transformations include:
    - DateTime conversion for timestamp columns
    - Categorical conversion for low-cardinality strings
    - Numeric validation and cleaning

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with type conversions applied

    Example:
        >>> df = _apply_type_conversions(df)
    """
    df_converted = df.copy()

    # Convert TransactionStartTime to datetime
    if 'TransactionStartTime' in df_converted.columns:
        try:
            df_converted['TransactionStartTime'] = pd.to_datetime(
                df_converted['TransactionStartTime'],
                errors='coerce'  # Invalid dates become NaT
            )
            logger.debug("✓ Converted TransactionStartTime to datetime64")
        except Exception as e:
            logger.warning(f"Failed to convert TransactionStartTime: {str(e)}")

    # Convert categorical columns
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy',
                       'CountryCode', 'ProviderId']
    for col in categorical_cols:
        if col in df_converted.columns and df_converted[col].dtype == 'object':
            df_converted[col] = df_converted[col].astype('category')

    # Ensure FraudResult is numeric
    if 'FraudResult' in df_converted.columns:
        try:
            df_converted['FraudResult'] = pd.to_numeric(
                df_converted['FraudResult'],
                errors='coerce'
            )
        except Exception as e:
            logger.warning(f"Failed to convert FraudResult to numeric: {str(e)}")

    logger.info("✓ Type conversions applied successfully")
    return df_converted


def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive data quality summary.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing quality metrics

    Example:
        >>> summary = get_data_quality_summary(df)
        >>> print(summary['completeness'])  # 0.95
    """
    summary = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'completeness': (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
        'duplicate_rows': df.duplicated().sum(),
        'unique_customers': df['AccountId'].nunique() if 'AccountId' in df.columns else None,
        'date_range': {
            'start': df['TransactionStartTime'].min().strftime('%Y-%m-%d')
            if 'TransactionStartTime' in df.columns and pd.notna(df['TransactionStartTime'].min())
            else 'Unknown',
            'end': df['TransactionStartTime'].max().strftime('%Y-%m-%d')
            if 'TransactionStartTime' in df.columns and pd.notna(df['TransactionStartTime'].max())
            else 'Unknown'
        }
    }

    logger.info(f"Data Quality Summary: {summary}")
    return summary


def split_train_test(df: pd.DataFrame, 
                     test_size: float = 0.2,
                     random_state: int = 42,
                     stratify_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets with stratification support.

    Args:
        df: Input DataFrame
        test_size: Proportion for test set (0.0-1.0)
        random_state: Random seed for reproducibility
        stratify_col: Column to stratify by (e.g., target variable)

    Returns:
        Tuple of (train_df, test_df)

    Raises:
        ValueError: If test_size invalid or stratify_col not found

    Example:
        >>> train_df, test_df = split_train_test(df, stratify_col='FraudResult')
    """
    # Defensive checks
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    if stratify_col and stratify_col not in df.columns:
        raise ValueError(f"Stratification column '{stratify_col}' not found")

    # Import here to avoid circular dependency
    from sklearn.model_selection import train_test_split

    if stratify_col:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[stratify_col]
        )
        logger.info(f"✓ Data split with stratification on '{stratify_col}'")
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )
        logger.info(f"✓ Data split without stratification")

    logger.info(f"  Train set: {train_df.shape[0]:,} rows")
    logger.info(f"  Test set: {test_df.shape[0]:,} rows")

    return train_df, test_df


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    print("Loading raw data...")
    df = load_raw_data(validate=True)
    print(f"\nData Quality Summary:")
    print(get_data_quality_summary(df))
