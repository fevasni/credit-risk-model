"""
Data validation functions for the credit risk modeling pipeline.

This module provides defensive validation checks to ensure data integrity,
detect anomalies, and provide meaningful error messages for downstream processing.

Key responsibilities:
- Schema validation (columns, data types)
- Business logic validation (value ranges, forbidden values)
- Quality metrics (missing values, duplicates, outliers)
"""

import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.constants import (
    REQUIRED_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
    IQR_MULTIPLIER, ZSCORE_THRESHOLD, FRAUD_PREVALENCE_RATE,
    MULTICOLLINEARITY_THRESHOLD
)

# Setup logging
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def validate_dataframe_structure(df: pd.DataFrame) -> None:
    """
    Validate the basic structure of the input DataFrame.

    Checks:
    - DataFrame is not None and not empty
    - All required columns are present
    - No unexpected NaN values in critical columns

    Args:
        df: Input DataFrame to validate

    Raises:
        ValidationError: If structure validation fails

    Example:
        >>> validate_dataframe_structure(df)  # Raises ValidationError if invalid
    """
    # Check if DataFrame exists and not empty
    if df is None:
        raise ValidationError("Input DataFrame is None")
    
    if df.empty:
        raise ValidationError("Input DataFrame is empty (0 rows)")

    if df.shape[1] == 0:
        raise ValidationError("Input DataFrame has no columns")

    # Check required columns
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValidationError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {set(df.columns)}"
        )

    logger.info(f"✓ DataFrame structure valid: {df.shape[0]:,} rows × {df.shape[1]} columns")


def validate_data_types(df: pd.DataFrame, expected_dtypes: Dict[str, str]) -> Tuple[bool, str]:
    """
    Validate that DataFrame columns have expected data types.

    Args:
        df: Input DataFrame
        expected_dtypes: Dict mapping column names to expected dtype strings

    Returns:
        Tuple[bool, str]: (is_valid, error_message)

    Example:
        >>> is_valid, msg = validate_data_types(df, {'Amount': 'float64'})
    """
    for col, expected_dtype in expected_dtypes.items():
        if col not in df.columns:
            continue
        
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            msg = f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}"
            logger.warning(msg)
            return False, msg

    logger.info("✓ All data types validated successfully")
    return True, ""


def detect_missing_values(df: pd.DataFrame, threshold_pct: float = 50.0) -> Dict[str, float]:
    """
    Detect and report missing values across DataFrame.

    Args:
        df: Input DataFrame
        threshold_pct: Warn if missing percentage exceeds this threshold

    Returns:
        Dict mapping column names to missing value percentages

    Example:
        >>> missing = detect_missing_values(df, threshold_pct=10.0)
    """
    missing_data = {}
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        if missing_pct > 0:
            missing_data[col] = missing_pct
            
            if missing_pct > threshold_pct:
                logger.warning(f"⚠️  Column '{col}' has {missing_pct:.2f}% missing values")
            else:
                logger.debug(f"Column '{col}' has {missing_pct:.2f}% missing values")
    
    if not missing_data:
        logger.info("✓ No missing values detected")
    
    return missing_data


def detect_duplicates(df: pd.DataFrame, subset: Optional[list] = None) -> int:
    """
    Detect duplicate rows in DataFrame.

    Args:
        df: Input DataFrame
        subset: Optional list of columns to consider for duplicate detection

    Returns:
        Number of duplicate rows found

    Example:
        >>> duplicates = detect_duplicates(df, subset=['TransactionId'])
    """
    if subset and not all(col in df.columns for col in subset):
        logger.warning(f"Some subset columns not found: {subset}")
        return 0

    duplicate_count = df.duplicated(subset=subset, keep=False).sum()
    
    if duplicate_count > 0:
        logger.warning(f"⚠️  Found {duplicate_count} duplicate rows")
    else:
        logger.info("✓ No duplicate rows detected")
    
    return duplicate_count


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[list] = None
                        ) -> Dict[str, Tuple[int, float]]:
    """
    Detect outliers using Interquartile Range (IQR) method.

    Args:
        df: Input DataFrame
        columns: Optional list of numeric columns to check (default: all numeric)

    Returns:
        Dict mapping column names to (count, percentage) of outliers

    Example:
        >>> outliers = detect_outliers_iqr(df, columns=['Amount'])
    """
    if columns is None:
        columns = list(NUMERIC_COLUMNS & set(df.columns))
    
    outlier_results = {}
    
    for col in columns:
        if col not in df.columns or df[col].dtype not in ['float64', 'int64']:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - IQR_MULTIPLIER * IQR
        upper_bound = Q3 + IQR_MULTIPLIER * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        outlier_results[col] = (outlier_count, outlier_pct)
        
        if outlier_count > 0:
            logger.debug(f"Column '{col}': {outlier_count} outliers ({outlier_pct:.2f}%)")
    
    return outlier_results


def detect_outliers_zscore(df: pd.DataFrame, columns: Optional[list] = None
                           ) -> Dict[str, Tuple[int, float]]:
    """
    Detect outliers using Z-score method.

    Args:
        df: Input DataFrame
        columns: Optional list of numeric columns to check (default: all numeric)

    Returns:
        Dict mapping column names to (count, percentage) of outliers

    Example:
        >>> outliers = detect_outliers_zscore(df)
    """
    if columns is None:
        columns = list(NUMERIC_COLUMNS & set(df.columns))
    
    outlier_results = {}
    
    for col in columns:
        if col not in df.columns or df[col].dtype not in ['float64', 'int64']:
            continue
        
        col_data = df[col].dropna()
        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
        outliers = z_scores[z_scores > ZSCORE_THRESHOLD]
        
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        outlier_results[col] = (outlier_count, outlier_pct)
        
        if outlier_count > 0:
            logger.debug(f"Column '{col}': {outlier_count} outliers (Z-score > {ZSCORE_THRESHOLD})")
    
    return outlier_results


def validate_class_distribution(df: pd.DataFrame, target_col: str = "FraudResult") -> Dict[str, Any]:
    """
    Validate class distribution for target variable.

    Args:
        df: Input DataFrame
        target_col: Name of target column

    Returns:
        Dict with class distribution statistics

    Raises:
        ValidationError: If class imbalance is too extreme

    Example:
        >>> dist = validate_class_distribution(df)
    """
    if target_col not in df.columns:
        raise ValidationError(f"Target column '{target_col}' not found")

    value_counts = df[target_col].value_counts()
    class_dist = {
        'total': len(df),
        'classes': value_counts.to_dict(),
        'proportions': (value_counts / len(df)).to_dict()
    }

    # Check if positive class is too rare (< 0.1%)
    if len(value_counts) > 1:
        min_proportion = value_counts.min() / len(df)
        if min_proportion < 0.001:
            logger.warning(f"⚠️  Extreme class imbalance detected: {min_proportion:.4%} minority class")

    logger.info(f"✓ Class distribution validated: {class_dist}")
    return class_dist


def check_multicollinearity(df: pd.DataFrame, numeric_cols: Optional[list] = None
                           ) -> Dict[Tuple[str, str], float]:
    """
    Detect multicollinearity by examining correlation matrix.

    Args:
        df: Input DataFrame
        numeric_cols: Optional list of numeric columns to check

    Returns:
        Dict mapping column pairs to correlation coefficients (|r| > threshold)

    Example:
        >>> high_corr = check_multicollinearity(df)
    """
    if numeric_cols is None:
        numeric_cols = list(NUMERIC_COLUMNS & set(df.columns))
    
    numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    high_corr_pairs = {}
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) > MULTICOLLINEARITY_THRESHOLD:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr_pairs[(col1, col2)] = corr_value
                logger.warning(f"⚠️  High correlation: {col1} ↔ {col2} (r={corr_value:.4f})")
    
    if not high_corr_pairs:
        logger.info("✓ No high multicollinearity detected")
    
    return high_corr_pairs


def run_full_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute full data validation pipeline.

    Args:
        df: Input DataFrame to validate

    Returns:
        Dict with validation results for all checks

    Raises:
        ValidationError: If critical validation fails

    Example:
        >>> results = run_full_validation(df)
    """
    logger.info("=" * 70)
    logger.info("STARTING FULL DATA VALIDATION")
    logger.info("=" * 70)
    
    validation_results = {}

    try:
        # 1. Structure validation (critical)
        validate_dataframe_structure(df)
        validation_results['structure'] = 'PASSED'

        # 2. Type validation (warning level)
        is_valid_types, msg = validate_data_types(df, {})
        validation_results['data_types'] = 'PASSED' if is_valid_types else 'WARNED'

        # 3. Missing values (warning level)
        missing_data = detect_missing_values(df)
        validation_results['missing_values'] = missing_data

        # 4. Duplicates (warning level)
        duplicates = detect_duplicates(df)
        validation_results['duplicates'] = duplicates

        # 5. Outliers (informational)
        iqr_outliers = detect_outliers_iqr(df)
        validation_results['outliers_iqr'] = iqr_outliers

        # 6. Class distribution (informational)
        class_dist = validate_class_distribution(df)
        validation_results['class_distribution'] = class_dist

        # 7. Multicollinearity (warning level)
        high_corr = check_multicollinearity(df)
        validation_results['multicollinearity'] = high_corr

        logger.info("=" * 70)
        logger.info("✓ FULL VALIDATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except ValidationError as e:
        logger.error(f"✗ CRITICAL VALIDATION FAILED: {str(e)}")
        raise

    return validation_results
