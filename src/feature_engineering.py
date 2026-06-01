"""
Feature engineering module for credit risk modeling.

This module provides modular, reusable functions for:
- RFM (Recency, Frequency, Monetary) customer aggregation
- Log transformations for skewed features
- Categorical encoding and binning
- Feature scaling and normalization

All functions include defensive checks and validation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
import numpy as np
from scipy.stats import skew

from src.constants import (
    RFM_RECENCY_WINDOWS_DAYS, RFM_AGGREGATION_LEVEL,
    LOG_TRANSFORM_COLUMNS, LOG_EPSILON, TOP_CATEGORIES_TO_KEEP,
    MULTICOLLINEARITY_THRESHOLD, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
)
from src.validators import ValidationError

# Setup logging
logger = logging.getLogger(__name__)


def compute_rfm_features(df: pd.DataFrame,
                        customer_col: str = "AccountId",
                        date_col: str = "TransactionStartTime",
                        amount_col: str = "Amount",
                        reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) features for customer segmentation.

    RFM analysis creates leading indicators of customer distress by measuring:
    - **Recency:** Days since last transaction (lower = more recent)
    - **Frequency:** Total transaction count over period
    - **Monetary:** Total or average transaction value

    Args:
        df: Input DataFrame with transactions
        customer_col: Column name for customer identifier
        date_col: Column name for transaction date/time
        amount_col: Column name for transaction amount
        reference_date: Reference date for recency calculation (default: max date in data)

    Returns:
        DataFrame with customer-level RFM features

    Raises:
        ValidationError: If required columns missing or data invalid

    Example:
        >>> rfm_df = compute_rfm_features(df, customer_col='AccountId')
        >>> rfm_df.head()
    """
    # Defensive checks
    required_cols = {customer_col, date_col, amount_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValidationError(f"Missing required columns: {missing_cols}")

    # Ensure date column is datetime
    if df[date_col].dtype != 'datetime64[ns]':
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        except Exception as e:
            raise ValidationError(f"Failed to convert {date_col} to datetime: {str(e)}")

    # Set reference date
    reference_date = reference_date or df[date_col].max()
    if pd.isna(reference_date):
        raise ValidationError(f"Cannot determine reference date from {date_col}")

    logger.info(f"Computing RFM features with reference date: {reference_date.date()}")

    # Group by customer and compute RFM
    rfm_df = df.groupby(customer_col).agg({
        date_col: [
            ('recency_days', lambda x: (reference_date - x.max()).days),
            ('last_transaction_date', 'max')
        ],
        'TransactionId': ('frequency', 'count'),  # Count of transactions
        amount_col: [
            ('monetary_total', 'sum'),
            ('monetary_mean', 'mean'),
            ('monetary_max', 'max'),
            ('monetary_std', 'std')
        ]
    }).reset_index()

    # Flatten multi-level columns
    rfm_df.columns = [col[0] if col[1] == '' else col[1] for col in rfm_df.columns]

    # Defensive check: non-empty result
    if len(rfm_df) == 0:
        raise ValidationError("RFM aggregation resulted in empty DataFrame")

    # Handle NaN in monetary_std (single transaction customers)
    rfm_df['monetary_std'] = rfm_df['monetary_std'].fillna(0)

    logger.info(f"✓ RFM computed for {len(rfm_df):,} unique customers")
    logger.info(f"  Recency range: {rfm_df['recency_days'].min()}-{rfm_df['recency_days'].max()} days")
    logger.info(f"  Frequency range: {rfm_df['frequency'].min()}-{rfm_df['frequency'].max()} transactions")
    logger.info(f"  Monetary range: {rfm_df['monetary_total'].min():.2f}-{rfm_df['monetary_total'].max():.2f}")

    return rfm_df


def apply_log_transformation(df: pd.DataFrame,
                             columns: Optional[List[str]] = None,
                             epsilon: float = LOG_EPSILON) -> pd.DataFrame:
    """
    Apply log transformation to right-skewed numeric columns.

    Log transformation stabilizes variance and improves linear model assumptions
    for highly skewed distributions (e.g., transaction amounts).

    Args:
        df: Input DataFrame
        columns: List of columns to transform (default: LOG_TRANSFORM_COLUMNS)
        epsilon: Offset for log(x + epsilon) to handle zeros/negatives

    Returns:
        DataFrame with log-transformed columns

    Raises:
        ValidationError: If columns not found or transformation fails

    Example:
        >>> df_transformed = apply_log_transformation(df, columns=['Amount'])
    """
    df_transformed = df.copy()
    columns_to_transform = columns or LOG_TRANSFORM_COLUMNS

    for col in columns_to_transform:
        if col not in df_transformed.columns:
            logger.warning(f"Column '{col}' not found, skipping log transformation")
            continue

        # Defensive check: numeric column
        if df_transformed[col].dtype not in ['float64', 'int64']:
            logger.warning(f"Column '{col}' is not numeric, skipping transformation")
            continue

        # Check for negative values
        if (df_transformed[col] < 0).any():
            logger.warning(f"Column '{col}' has negative values, applying absolute value first")
            df_transformed[col] = df_transformed[col].abs()

        # Compute skewness before transformation
        skewness_before = skew(df_transformed[col].dropna())

        # Apply log transformation
        df_transformed[f'{col}_log'] = np.log(df_transformed[col] + epsilon)

        # Compute skewness after transformation
        skewness_after = skew(df_transformed[f'{col}_log'].dropna())

        logger.info(
            f"✓ Log transformation applied to '{col}': "
            f"skewness {skewness_before:.4f} → {skewness_after:.4f}"
        )

    return df_transformed


def bin_categorical_features(df: pd.DataFrame,
                              categorical_cols: Optional[Dict[str, int]] = None,
                              min_frequency: int = 10) -> pd.DataFrame:
    """
    Bin low-frequency categorical values into "Other" category.

    Reduces feature dimensionality and prevents sparse encoding while preserving
    information about common categories.

    Args:
        df: Input DataFrame
        categorical_cols: Dict mapping column names to max categories to keep
                         (default: TOP_CATEGORIES_TO_KEEP)
        min_frequency: Minimum frequency threshold for keeping a category

    Returns:
        DataFrame with binned categorical features

    Example:
        >>> df_binned = bin_categorical_features(df, {'ProductCategory': 5})
    """
    df_binned = df.copy()
    categorical_cols = categorical_cols or TOP_CATEGORIES_TO_KEEP

    for col, max_categories in categorical_cols.items():
        if col not in df_binned.columns:
            continue

        # Count value frequencies
        value_counts = df_binned[col].value_counts()

        # Determine categories to keep
        keep_categories = set(value_counts.head(max_categories).index)

        # Bin infrequent categories
        df_binned[col] = df_binned[col].apply(
            lambda x: x if x in keep_categories else 'Other'
        )

        num_unique = df_binned[col].nunique()
        logger.info(f"✓ Binned '{col}': {len(value_counts)} → {num_unique} categories")

    return df_binned


def encode_categorical_onehot(df: pd.DataFrame,
                              columns: Optional[List[str]] = None,
                              drop_first: bool = True) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical columns.

    Creates binary features for categorical values, enabling use in
    linear models. Handles missing values defensively.

    Args:
        df: Input DataFrame
        columns: List of columns to encode (default: CATEGORICAL_COLUMNS)
        drop_first: Whether to drop first category (prevents multicollinearity)

    Returns:
        DataFrame with one-hot encoded categorical features

    Example:
        >>> df_encoded = encode_categorical_onehot(df)
    """
    df_encoded = df.copy()
    columns_to_encode = columns or list(CATEGORICAL_COLUMNS & set(df.columns))

    for col in columns_to_encode:
        if col not in df_encoded.columns:
            continue

        try:
            # Apply one-hot encoding with NaN handling
            encoded = pd.get_dummies(
                df_encoded[col],
                prefix=col,
                drop_first=drop_first,
                dummy_na=True  # Create column for NaN values
            )

            # Concatenate encoded columns
            df_encoded = pd.concat([df_encoded, encoded], axis=1)

            # Drop original column
            df_encoded = df_encoded.drop(col, axis=1)

            num_new_features = encoded.shape[1]
            logger.info(f"✓ One-hot encoded '{col}': created {num_new_features} binary features")

        except Exception as e:
            logger.warning(f"Failed to encode '{col}': {str(e)}")

    return df_encoded


def scale_numeric_features(df: pd.DataFrame,
                          method: str = 'standard',
                          columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Scale numeric features using standardization or normalization.

    Scaling prevents features with larger magnitudes from dominating
    linear models.

    Args:
        df: Input DataFrame
        method: 'standard' (mean=0, std=1) or 'minmax' (0-1 range)
        columns: Specific columns to scale (default: all numeric)

    Returns:
        Tuple of (scaled_df, scaling_params dict for inverse transformation)

    Example:
        >>> df_scaled, params = scale_numeric_features(df, method='standard')
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    df_scaled = df.copy()
    columns_to_scale = columns or list(NUMERIC_COLUMNS & set(df.columns))

    scaling_params = {'method': method, 'columns': columns_to_scale, 'scalers': {}}

    scaler_class = StandardScaler if method == 'standard' else MinMaxScaler

    for col in columns_to_scale:
        if col not in df_scaled.columns or df_scaled[col].dtype not in ['float64', 'int64']:
            continue

        # Fit scaler
        scaler = scaler_class()
        values = df_scaled[col].values.reshape(-1, 1)

        try:
            scaled_values = scaler.fit_transform(values)
            df_scaled[col] = scaled_values.flatten()
            scaling_params['scalers'][col] = scaler

            logger.debug(f"✓ Scaled '{col}' using {method} method")

        except Exception as e:
            logger.warning(f"Failed to scale '{col}': {str(e)}")

    logger.info(f"✓ Scaled {len([c for c in columns_to_scale if c in df_scaled.columns])} numeric features")
    return df_scaled, scaling_params


def remove_multicollinear_features(df: pd.DataFrame,
                                  threshold: float = MULTICOLLINEARITY_THRESHOLD
                                  ) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Identify and remove highly correlated features to prevent multicollinearity.

    Args:
        df: Input DataFrame with numeric features
        threshold: Correlation coefficient threshold (default: 0.7)

    Returns:
        Tuple of (df_reduced, removed_columns set)

    Example:
        >>> df_reduced, removed = remove_multicollinear_features(df, threshold=0.8)
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        logger.info("✓ No multicollinearity check needed (< 2 numeric features)")
        return df, set()

    # Compute correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Identify features to drop
    features_to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                col_to_drop = corr_matrix.columns[j]  # Drop second of pair
                features_to_drop.add(col_to_drop)

                logger.warning(
                    f"⚠️  Dropping '{col_to_drop}' (corr with "
                    f"'{corr_matrix.columns[i]}' = {corr_matrix.iloc[i, j]:.4f})"
                )

    # Drop features
    df_reduced = df.drop(columns=list(features_to_drop & set(df.columns)))

    logger.info(f"✓ Removed {len(features_to_drop)} multicollinear features")
    return df_reduced, features_to_drop


def create_interaction_features(df: pd.DataFrame,
                               feature_pairs: Optional[List[Tuple[str, str]]] = None
                               ) -> pd.DataFrame:
    """
    Create polynomial/interaction features from numeric columns.

    Interaction terms capture non-linear relationships that linear models
    cannot express independently.

    Args:
        df: Input DataFrame
        feature_pairs: Optional list of (col1, col2) pairs to create interactions

    Returns:
        DataFrame with interaction features added

    Example:
        >>> df_interact = create_interaction_features(df, 
        ...                     feature_pairs=[('monetary_total', 'frequency')])
    """
    df_interact = df.copy()

    if feature_pairs is None:
        # Create interactions for all numeric pairs
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_pairs = [(numeric_cols[i], numeric_cols[j])
                        for i in range(len(numeric_cols))
                        for j in range(i + 1, min(i + 3, len(numeric_cols)))]

    for col1, col2 in feature_pairs:
        if col1 not in df_interact.columns or col2 not in df_interact.columns:
            continue

        try:
            interaction_col = f'{col1}_x_{col2}'
            df_interact[interaction_col] = df_interact[col1] * df_interact[col2]
            logger.debug(f"✓ Created interaction feature: {interaction_col}")

        except Exception as e:
            logger.warning(f"Failed to create interaction {col1} × {col2}: {str(e)}")

    return df_interact


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage documentation
    print("""
    Feature Engineering Module - Example Usage:
    
    1. Compute RFM features:
       >>> rfm_df = compute_rfm_features(df)
    
    2. Apply log transformation:
       >>> df_transformed = apply_log_transformation(df, columns=['Amount'])
    
    3. Bin categorical features:
       >>> df_binned = bin_categorical_features(df)
    
    4. Encode categorical features:
       >>> df_encoded = encode_categorical_onehot(df_binned)
    
    5. Scale numeric features:
       >>> df_scaled, params = scale_numeric_features(df_encoded)
    
    6. Remove multicollinear features:
       >>> df_final, removed = remove_multicollinear_features(df_scaled)
    """)
