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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
        'TransactionId': [
            ('frequency', 'count')  # Count of transactions
        ],
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


def cluster_customers_kmeans(rfm_df: pd.DataFrame,
                             n_clusters: int = 3,
                             rfm_columns: Optional[List[str]] = None,
                             random_state: int = 42,
                             customer_col: str = "AccountId") -> Tuple[pd.DataFrame, dict]:
    """
    Segment customers into clusters using K-Means on RFM features.

    Performs feature scaling (standardization) before clustering to ensure all
    RFM dimensions contribute equally to distance calculations. Sets random_state
    for reproducibility.

    Args:
        rfm_df: DataFrame with RFM features (output from compute_rfm_features)
        n_clusters: Number of customer segments (default: 3)
        rfm_columns: List of RFM columns to use for clustering.
                    Default: ['recency_days', 'frequency', 'monetary_total']
        random_state: Random seed for reproducibility (default: 42)
        customer_col: Name of customer identifier column (default: 'AccountId')

    Returns:
        Tuple of:
        - DataFrame with added 'cluster' column
        - Dictionary with cluster metadata (scaler, model, feature names)

    Raises:
        ValidationError: If required columns missing or clustering fails

    Example:
        >>> rfm_df = compute_rfm_features(df)
        >>> clustered_df, metadata = cluster_customers_kmeans(rfm_df, n_clusters=3)
        >>> clustered_df.head()
    """
    # Defensive checks
    if customer_col not in rfm_df.columns:
        raise ValidationError(f"Customer column '{customer_col}' not found")

    # Define RFM columns
    if rfm_columns is None:
        rfm_columns = ['recency_days', 'frequency', 'monetary_total']

    missing_cols = set(rfm_columns) - set(rfm_df.columns)
    if missing_cols:
        raise ValidationError(f"Missing RFM columns: {missing_cols}")

    logger.info(f"Clustering customers into {n_clusters} segments using RFM features")

    # Prepare data for clustering
    X = rfm_df[rfm_columns].copy()

    # Handle missing values
    if X.isna().any().any():
        logger.warning("Missing values detected in RFM features, filling with 0")
        X = X.fillna(0)

    # Standardize features (critical for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"  Features scaled: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

    # Perform K-Means clustering with fixed random_state
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    logger.info(f"✓ K-Means clustering completed with random_state={random_state}")

    # Create result DataFrame
    result_df = rfm_df.copy()
    result_df['cluster'] = clusters

    # Log cluster statistics
    for cluster_id in range(n_clusters):
        cluster_mask = result_df['cluster'] == cluster_id
        n_customers = cluster_mask.sum()
        
        logger.info(f"  Cluster {cluster_id}: {n_customers:,} customers")
        logger.info(f"    Recency (days): {result_df.loc[cluster_mask, 'recency_days'].mean():.1f}")
        logger.info(f"    Frequency: {result_df.loc[cluster_mask, 'frequency'].mean():.1f}")
        logger.info(f"    Monetary: {result_df.loc[cluster_mask, 'monetary_total'].mean():.2f}")

    # Store metadata for future use
    metadata = {
        'scaler': scaler,
        'kmeans_model': kmeans,
        'rfm_columns': rfm_columns,
        'n_clusters': n_clusters,
        'random_state': random_state,
        'cluster_centers_scaled': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_
    }

    return result_df, metadata


def identify_high_risk_cluster(clustered_df: pd.DataFrame,
                               cluster_col: str = 'cluster',
                               frequency_col: str = 'frequency',
                               monetary_col: str = 'monetary_total') -> int:
    """
    Identify the high-risk customer cluster based on low engagement metrics.

    The high-risk cluster is characterized by:
    - Low frequency (fewest transactions)
    - Low monetary value (smallest transaction amounts)

    Args:
        clustered_df: DataFrame with cluster assignments (from cluster_customers_kmeans)
        cluster_col: Name of cluster assignment column (default: 'cluster')
        frequency_col: Name of frequency column (default: 'frequency')
        monetary_col: Name of monetary value column (default: 'monetary_total')

    Returns:
        Cluster ID representing the high-risk segment

    Raises:
        ValidationError: If required columns missing or logic fails

    Example:
        >>> high_risk_cluster = identify_high_risk_cluster(clustered_df)
        >>> print(f"High-risk cluster: {high_risk_cluster}")
    """
    if cluster_col not in clustered_df.columns:
        raise ValidationError(f"Cluster column '{cluster_col}' not found")

    if frequency_col not in clustered_df.columns:
        raise ValidationError(f"Frequency column '{frequency_col}' not found")

    if monetary_col not in clustered_df.columns:
        raise ValidationError(f"Monetary column '{monetary_col}' not found")

    logger.info("Identifying high-risk cluster (low frequency & low monetary value)...")

    # Compute cluster-level statistics
    cluster_stats = clustered_df.groupby(cluster_col).agg({
        frequency_col: ['mean', 'std', 'min', 'max'],
        monetary_col: ['mean', 'std', 'min', 'max']
    }).round(2)

    logger.info("Cluster statistics:")
    logger.info(cluster_stats)

    # Find cluster with lowest average frequency AND lowest average monetary value
    cluster_means = clustered_df.groupby(cluster_col).agg({
        frequency_col: 'mean',
        monetary_col: 'mean'
    })

    # Normalize scores for combined evaluation
    freq_normalized = (cluster_means[frequency_col] - cluster_means[frequency_col].min()) / \
                      (cluster_means[frequency_col].max() - cluster_means[frequency_col].min())
    monetary_normalized = (cluster_means[monetary_col] - cluster_means[monetary_col].min()) / \
                         (cluster_means[monetary_col].max() - cluster_means[monetary_col].min())

    # Combined risk score (lower = higher risk)
    risk_score = freq_normalized + monetary_normalized

    high_risk_cluster_id = risk_score.idxmin()

    logger.info(f"✓ High-risk cluster identified: Cluster {high_risk_cluster_id}")
    logger.info(f"  Average frequency: {cluster_means.loc[high_risk_cluster_id, frequency_col]:.1f}")
    logger.info(f"  Average monetary: {cluster_means.loc[high_risk_cluster_id, monetary_col]:.2f}")
    logger.info(f"  Risk score: {risk_score.loc[high_risk_cluster_id]:.4f}")

    return int(high_risk_cluster_id)


def assign_high_risk_labels(rfm_df: pd.DataFrame,
                           n_clusters: int = 3,
                           random_state: int = 42,
                           customer_col: str = "AccountId") -> Tuple[pd.DataFrame, int]:
    """
    Complete pipeline: cluster customers and assign high-risk binary labels.

    This function orchestrates:
    1. K-Means clustering of RFM features
    2. High-risk cluster identification
    3. Binary label assignment (is_high_risk: 1 for high-risk, 0 otherwise)

    Args:
        rfm_df: DataFrame with RFM features (output from compute_rfm_features)
        n_clusters: Number of clusters for K-Means (default: 3)
        random_state: Random seed for reproducibility (default: 42)
        customer_col: Name of customer identifier column

    Returns:
        Tuple of:
        - DataFrame with new 'is_high_risk' binary column (1 or 0)
        - High-risk cluster ID used for labeling

    Example:
        >>> rfm_df = compute_rfm_features(transactions_df)
        >>> labeled_df, high_risk_id = assign_high_risk_labels(rfm_df)
        >>> print(f"High-risk customers: {labeled_df['is_high_risk'].sum()}")
    """
    logger.info("="*70)
    logger.info("RFM-Based High-Risk Customer Labeling Pipeline")
    logger.info("="*70)

    # Step 1: Cluster customers
    clustered_df, cluster_metadata = cluster_customers_kmeans(
        rfm_df,
        n_clusters=n_clusters,
        random_state=random_state,
        customer_col=customer_col
    )

    # Step 2: Identify high-risk cluster
    high_risk_cluster_id = identify_high_risk_cluster(clustered_df)

    # Step 3: Assign binary labels
    labeled_df = clustered_df.copy()
    labeled_df['is_high_risk'] = (labeled_df['cluster'] == high_risk_cluster_id).astype(int)

    # Log summary
    n_high_risk = labeled_df['is_high_risk'].sum()
    n_total = len(labeled_df)
    pct_high_risk = (n_high_risk / n_total) * 100

    logger.info("="*70)
    logger.info("High-Risk Label Assignment Summary")
    logger.info("="*70)
    logger.info(f"✓ Total customers: {n_total:,}")
    logger.info(f"✓ High-risk customers: {n_high_risk:,} ({pct_high_risk:.2f}%)")
    logger.info(f"✓ Low-risk customers: {n_total - n_high_risk:,} ({100-pct_high_risk:.2f}%)")
    logger.info(f"✓ High-risk cluster ID: {high_risk_cluster_id}")

    return labeled_df, high_risk_cluster_id


def integrate_high_risk_target(transaction_df: pd.DataFrame,
                              rfm_df: pd.DataFrame,
                              customer_col: str = "AccountId") -> pd.DataFrame:
    """
    Merge high-risk labels back into the main transaction dataset.

    This function takes customer-level high-risk labels and joins them
    back to the transaction-level dataset, creating a binary target column
    for model training.

    Args:
        transaction_df: Original transaction-level DataFrame
        rfm_df: Customer-level DataFrame with 'is_high_risk' column
                (output from assign_high_risk_labels)
        customer_col: Name of customer identifier column

    Returns:
        Transaction-level DataFrame with added 'is_high_risk' column

    Raises:
        ValidationError: If merge fails or columns missing

    Example:
        >>> labeled_df, _ = assign_high_risk_labels(rfm_df)
        >>> transactions_with_target = integrate_high_risk_target(df, labeled_df)
        >>> print(transactions_with_target[['AccountId', 'is_high_risk']].head())
    """
    if 'is_high_risk' not in rfm_df.columns:
        raise ValidationError("'is_high_risk' column not found in RFM DataFrame")

    if customer_col not in transaction_df.columns:
        raise ValidationError(f"Customer column '{customer_col}' not found in transaction data")

    if customer_col not in rfm_df.columns:
        raise ValidationError(f"Customer column '{customer_col}' not found in RFM data")

    logger.info("Merging high-risk labels back into transaction data...")

    # Select only customer_col and is_high_risk for merge
    customer_labels = rfm_df[[customer_col, 'is_high_risk']].drop_duplicates()

    # Merge on customer identifier
    result_df = transaction_df.merge(
        customer_labels,
        on=customer_col,
        how='left'
    )

    # Defensive check: verify merge success
    if result_df['is_high_risk'].isna().any():
        logger.warning("⚠️  Some transactions have missing is_high_risk values after merge")
        # Fill any remaining NaNs with 0 (not high-risk)
        result_df['is_high_risk'] = result_df['is_high_risk'].fillna(0).astype(int)

    # Log merge results
    n_transactions = len(result_df)
    n_high_risk_transactions = (result_df['is_high_risk'] == 1).sum()
    pct_high_risk = (n_high_risk_transactions / n_transactions) * 100

    logger.info(f"✓ Merge completed successfully")
    logger.info(f"  Total transactions: {n_transactions:,}")
    logger.info(f"  From high-risk customers: {n_high_risk_transactions:,} ({pct_high_risk:.2f}%)")
    logger.info(f"  From low-risk customers: {n_transactions - n_high_risk_transactions:,} ({100-pct_high_risk:.2f}%)")

    # Verify data integrity
    if result_df.shape[0] != transaction_df.shape[0]:
        logger.warning("⚠️  Row count changed after merge (possible duplicates)")

    return result_df


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
    
    2. Cluster customers and assign high-risk labels:
       >>> labeled_df, high_risk_cluster_id = assign_high_risk_labels(rfm_df)
    
    3. Integrate labels into transaction data:
       >>> df_with_target = integrate_high_risk_target(df, labeled_df)
    
    4. Apply log transformation:
       >>> df_transformed = apply_log_transformation(df, columns=['Amount'])
    
    5. Bin categorical features:
       >>> df_binned = bin_categorical_features(df)
    
    6. Encode categorical features:
       >>> df_encoded = encode_categorical_onehot(df_binned)
    
    7. Scale numeric features:
       >>> df_scaled, params = scale_numeric_features(df_encoded)
    
    8. Remove multicollinear features:
       >>> df_final, removed = remove_multicollinear_features(df_scaled)
    """)
