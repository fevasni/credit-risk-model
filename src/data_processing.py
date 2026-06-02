"""
Comprehensive Data Processing and Feature Engineering Pipeline for Credit Risk Modeling.

This module implements a complete, production-ready pipeline that transforms raw transaction
data into model-ready features. The pipeline is designed as a single sklearn Pipeline object
that handles:

1. Aggregate Features: Customer-level aggregations (sum, mean, count, std of transactions)
2. Temporal Features: Extract hour, day, month, year from transaction timestamps
3. Categorical Encoding: One-Hot Encoding and Label Encoding
4. Missing Value Handling: Imputation strategies for missing values
5. Normalization/Standardization: Scale numerical features to comparable ranges
6. Weight of Evidence (WoE) and Information Value (IV): Statistical encoding for predictive power

Key Responsibilities:
- Load and validate raw transaction data
- Engineer customer-level and transaction-level features
- Handle categorical variables through multiple encoding strategies
- Apply advanced statistical transformations (WoE/IV)
- Create a fitted, reusable pipeline for model training and prediction
- Provide comprehensive logging and error handling

Author: Credit Risk Modeling Team
Date: 2026-06-02
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import warnings

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import category_encoders as ce

from src.constants import (
    RAW_DATA_FILE, DATA_DICTIONARY_FILE,
    NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, REQUIRED_COLUMNS,
    TOP_CATEGORIES_TO_KEEP, LOG_TRANSFORM_COLUMNS, LOG_EPSILON,
    IQR_MULTIPLIER, ZSCORE_THRESHOLD
)
from src.data_loader import load_raw_data, load_data_dictionary
from src.validators import ValidationError, validate_dataframe_structure

# Setup logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# Custom Transformers for Feature Engineering
# ============================================================================

class AggregateFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates customer-level aggregate features from transaction-level data.
    
    Features created:
    - total_transactions: Count of transactions per customer
    - total_amount: Sum of all transaction amounts per customer
    - mean_amount: Average transaction amount per customer
    - std_amount: Standard deviation of transaction amounts per customer
    - max_amount: Maximum transaction amount per customer
    - min_amount: Minimum transaction amount per customer
    - days_since_first_transaction: Days between first and last transaction
    - recency_days: Days since last transaction
    - fraud_count: Count of fraudulent transactions per customer
    - fraud_rate: Fraud rate per customer
    - night_transaction_ratio: Ratio of night transactions
    - weekend_transaction_ratio: Ratio of weekend transactions
    - most_common_hour: Most frequent transaction hour
    - most_common_day: Most frequent transaction day of month
    """
    
    def __init__(self, 
                 customer_col: str = "AccountId",
                 date_col: str = "TransactionStartTime",
                 amount_col: str = "Amount",
                 fraud_col: str = "FraudResult",
                 reference_date: Optional[pd.Timestamp] = None):
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.fraud_col = fraud_col
        self.reference_date = reference_date
        self.agg_features_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the aggregator (learning statistics from training data)."""
        if self.date_col not in X.columns:
            raise ValidationError(f"Column '{self.date_col}' not found in data")
        
        # Convert date column to datetime if not already
        X_copy = X.copy()
        if X_copy[self.date_col].dtype != 'datetime64[ns]':
            X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col], errors='coerce')
        
        # Set reference date for recency calculation
        self.reference_date = self.reference_date or X_copy[self.date_col].max()
        
        logger.info(f"AggregateFeatureTransformer fitted with {len(X)} transactions")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform transaction-level data to customer-level aggregates."""
        X_copy = X.copy()
        
        # Convert date column if necessary
        if X_copy[self.date_col].dtype != 'datetime64[ns]':
            X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col], errors='coerce')
        
        # Extract temporal features at transaction level
        X_copy['transaction_hour'] = X_copy[self.date_col].dt.hour
        X_copy['transaction_day'] = X_copy[self.date_col].dt.day
        X_copy['is_night_transaction'] = (
            (X_copy['transaction_hour'] >= 22) | (X_copy['transaction_hour'] < 5)
        ).astype(int)
        X_copy['is_weekend'] = (X_copy[self.date_col].dt.dayofweek >= 5).astype(int)
        
        # Compute aggregations
        agg_dict = {
            'TransactionId': 'count',  # frequency
            self.amount_col: ['sum', 'mean', 'std', 'max', 'min'],
            self.date_col: ['min', 'max'],  # for recency and duration
            self.fraud_col: ['sum', 'mean'],  # fraud statistics
            'is_night_transaction': 'sum',  # count of night transactions
            'is_weekend': 'sum',  # count of weekend transactions
            'transaction_hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),  # most common hour
            'transaction_day': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),   # most common day
        }
        
        agg_df = X_copy.groupby(self.customer_col).agg(agg_dict).reset_index()
        
        # Flatten multi-level columns
        agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in agg_df.columns]
        
        # Rename columns for clarity
        rename_map = {
            'TransactionId_count': 'total_transactions',
            f'{self.amount_col}_sum': 'total_amount',
            f'{self.amount_col}_mean': 'mean_amount',
            f'{self.amount_col}_std': 'std_amount',
            f'{self.amount_col}_max': 'max_amount',
            f'{self.amount_col}_min': 'min_amount',
            f'{self.date_col}_min': 'first_transaction_date',
            f'{self.date_col}_max': 'last_transaction_date',
            f'{self.fraud_col}_sum': 'fraud_count',
            f'{self.fraud_col}_mean': 'fraud_rate',
            'is_night_transaction_sum': 'night_transaction_count',
            'is_weekend_sum': 'weekend_transaction_count',
        }
        
        # Only rename columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in agg_df.columns}
        agg_df = agg_df.rename(columns=rename_map)
        
        # Calculate derived features
        agg_df['days_since_first_tx'] = (
            agg_df['last_transaction_date'] - agg_df['first_transaction_date']
        ).dt.days
        
        agg_df['recency_days'] = (
            self.reference_date - agg_df['last_transaction_date']
        ).dt.days
        
        # Calculate ratios
        agg_df['night_transaction_ratio'] = (
            agg_df['night_transaction_count'] / agg_df['total_transactions']
        ).fillna(0)
        
        agg_df['weekend_transaction_ratio'] = (
            agg_df['weekend_transaction_count'] / agg_df['total_transactions']
        ).fillna(0)
        
        # Handle NaN in std_amount (occurs when customer has single transaction)
        agg_df['std_amount'] = agg_df['std_amount'].fillna(0)
        
        # Drop date columns and transaction counts as they're not needed for modeling
        drop_cols = ['first_transaction_date', 'last_transaction_date', 
                    'night_transaction_count', 'weekend_transaction_count',
                    'transaction_hour', 'transaction_day']
        agg_df = agg_df.drop(
            columns=[col for col in drop_cols if col in agg_df.columns],
            errors='ignore'
        )
        
        logger.info(f"Generated {len(agg_df)} customer-level aggregate records")
        return agg_df


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features from transaction timestamps.
    
    Features created:
    - transaction_hour: Hour of day (0-23)
    - transaction_day: Day of month (1-31)
    - transaction_dow: Day of week (0=Monday, 6=Sunday)
    - transaction_month: Month (1-12)
    - transaction_quarter: Quarter (1-4)
    - transaction_year: Year
    - is_weekend: Binary indicator for weekend transactions
    - is_night_transaction: Binary indicator for night hours (22:00-05:00)
    """
    
    def __init__(self, date_col: str = "TransactionStartTime"):
        self.date_col = date_col
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (no fitting needed for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform transaction timestamps into temporal features."""
        X_copy = X.copy()
        
        # Convert date column if necessary
        if X_copy[self.date_col].dtype != 'datetime64[ns]':
            X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col], errors='coerce')
        
        # Extract temporal features
        X_copy['transaction_hour'] = X_copy[self.date_col].dt.hour
        X_copy['transaction_day'] = X_copy[self.date_col].dt.day
        X_copy['transaction_dow'] = X_copy[self.date_col].dt.dayofweek
        X_copy['transaction_month'] = X_copy[self.date_col].dt.month
        X_copy['transaction_quarter'] = X_copy[self.date_col].dt.quarter
        X_copy['transaction_year'] = X_copy[self.date_col].dt.year
        
        # Derived binary features
        X_copy['is_weekend'] = (X_copy['transaction_dow'] >= 5).astype(int)
        X_copy['is_night_transaction'] = (
            (X_copy['transaction_hour'] >= 22) | 
            (X_copy['transaction_hour'] < 5)
        ).astype(int)
        
        logger.debug(f"Generated temporal features for {len(X_copy)} transactions")
        return X_copy


class CategoricalBinningTransformer(BaseEstimator, TransformerMixin):
    """
    Bins categorical variables by retaining top N categories and grouping rest as 'Other'.
    
    Helps reduce dimensionality and avoid sparse categories that don't provide signal.
    """
    
    def __init__(self, top_categories: Dict[str, int] = None):
        self.top_categories = top_categories or TOP_CATEGORIES_TO_KEEP
        self.top_cats_per_col_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn the top categories for each categorical column."""
        for col, n_top in self.top_categories.items():
            if col in X.columns:
                # Get top N categories by frequency
                top_cats = X[col].value_counts().head(n_top).index.tolist()
                self.top_cats_per_col_[col] = top_cats
                logger.debug(f"Top {n_top} categories for '{col}': {top_cats[:3]}...")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply binning: keep top categories, group rest as 'Other'."""
        X_copy = X.copy()
        
        for col, top_cats in self.top_cats_per_col_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].apply(
                    lambda x: x if x in top_cats else 'Other'
                )
        
        logger.debug(f"Applied categorical binning to {len(self.top_cats_per_col_)} columns")
        return X_copy


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handles missing values in the dataset using multiple strategies.
    
    Strategies:
    - Numeric columns: Fill with mean (or median/mode)
    - Categorical columns: Fill with mode (most frequent value)
    - Rows with critical missing values: Remove
    """
    
    def __init__(self, 
                 numeric_strategy: str = 'mean',
                 categorical_strategy: str = 'most_frequent',
                 critical_cols: Set[str] = None):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.critical_cols = critical_cols or set()
        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn imputation statistics from training data."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_cols:
            self.numeric_imputer_ = SimpleImputer(strategy=self.numeric_strategy)
            self.numeric_imputer_.fit(X[numeric_cols])
        
        if categorical_cols:
            self.categorical_imputer_ = SimpleImputer(strategy=self.categorical_strategy)
            self.categorical_imputer_.fit(X[categorical_cols])
        
        logger.info(f"Fitted imputers for {len(numeric_cols)} numeric and "
                   f"{len(categorical_cols)} categorical columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation to handle missing values."""
        X_copy = X.copy()
        
        # Remove rows with missing critical columns (if any exist)
        critical_cols_present = [col for col in self.critical_cols if col in X_copy.columns]
        if critical_cols_present:
            X_copy = X_copy.dropna(subset=critical_cols_present, how='any')
        
        # Impute missing values in numeric columns
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols and self.numeric_imputer_:
            X_copy[numeric_cols] = self.numeric_imputer_.transform(X_copy[numeric_cols])
        
        # Impute missing values in categorical columns
        categorical_cols = X_copy.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols and self.categorical_imputer_:
            X_copy[categorical_cols] = self.categorical_imputer_.transform(X_copy[categorical_cols])
        
        logger.debug(f"Applied missing value imputation to {len(X_copy)} records")
        return X_copy


class NumericalScalingTransformer(BaseEstimator, TransformerMixin):
    """
    Scales only numerical columns in the DataFrame, leaving categorical columns unchanged.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler_type = scaler_type
        self.scaler_ = None
        self.numeric_cols_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn scaling parameters from numeric columns."""
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.numeric_cols_:
            if self.scaler_type == 'standard':
                self.scaler_ = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler_ = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler: {self.scaler_type}")
            
            self.scaler_.fit(X[self.numeric_cols_])
        
        logger.debug(f"Fitted {self.scaler_type} scaler for {len(self.numeric_cols_)} numeric columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to numeric columns only."""
        X_copy = X.copy()
        
        if self.numeric_cols_ and self.scaler_:
            X_copy[self.numeric_cols_] = self.scaler_.transform(X_copy[self.numeric_cols_])
        
        logger.debug(f"Applied {self.scaler_type} scaling to {len(self.numeric_cols_)} columns")
        return X_copy


class OutlierRemovalTransformer(BaseEstimator, TransformerMixin):
    """
    Detects and removes or caps outliers using IQR or Z-score methods.
    """
    
    def __init__(self, 
                 method: str = 'iqr',
                 iqr_multiplier: float = IQR_MULTIPLIER,
                 zscore_threshold: float = ZSCORE_THRESHOLD):
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.bounds_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn outlier bounds from training data."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                self.bounds_[col] = (lower_bound, upper_bound)
            
            elif self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                lower_bound = mean - self.zscore_threshold * std
                upper_bound = mean + self.zscore_threshold * std
                self.bounds_[col] = (lower_bound, upper_bound)
        
        logger.info(f"Fitted outlier bounds for {len(self.bounds_)} numeric columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier removal/capping."""
        X_copy = X.copy()
        
        outlier_count = 0
        for col, (lower, upper) in self.bounds_.items():
            if col in X_copy.columns:
                mask = (X_copy[col] < lower) | (X_copy[col] > upper)
                outlier_count += mask.sum()
                
                # Cap outliers instead of removing them
                X_copy[col] = X_copy[col].clip(lower, upper)
        
        logger.debug(f"Handled {outlier_count} outlier values")
        return X_copy


class LogTransformTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log transformation to reduce skewness in features.
    
    Useful for highly skewed distributions (e.g., transaction amounts).
    Formula: log(x + epsilon) where epsilon prevents log(0) errors.
    """
    
    def __init__(self, 
                 log_cols: List[str] = None,
                 epsilon: float = LOG_EPSILON):
        self.log_cols = log_cols
        self.epsilon = epsilon
        self.cols_to_transform_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Identify which log columns actually exist in the data."""
        # If no specific columns specified, use default
        if self.log_cols is None:
            cols_to_check = LOG_TRANSFORM_COLUMNS
        else:
            cols_to_check = self.log_cols
        
        # Only transform columns that exist and are numeric
        self.cols_to_transform_ = [
            col for col in cols_to_check
            if col in X.columns and X[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        
        logger.debug(f"LogTransformTransformer will transform: {self.cols_to_transform_}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to specified columns."""
        X_copy = X.copy()
        
        for col in self.cols_to_transform_:
            if col in X_copy.columns:
                # Handle negative values and zeros
                X_copy[f'{col}_log'] = np.log1p(X_copy[col].abs() + self.epsilon)
        
        logger.debug(f"Applied log transformation to {len(self.cols_to_transform_)} columns")
        return X_copy


class WoEIVTransformer(BaseEstimator, TransformerMixin):
    """
    Computes Weight of Evidence (WoE) and Information Value (IV) for categorical features.
    
    WoE is a statistical technique that measures the strength of a feature's relationship
    with the target variable. It transforms categorical variables into continuous values.
    
    Formula:
    WoE = ln(% of Events / % of Non-Events)
    IV = Sum[(% of Events - % of Non-Events) * WoE]
    
    Higher IV values indicate stronger predictive power.
    """
    
    def __init__(self, 
                 categorical_cols: List[str] = None,
                 target_col: str = 'FraudResult',
                 min_frequency: int = 5):
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.min_frequency = min_frequency
        self.woe_dicts_ = {}
        self.iv_scores_ = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Compute WoE transformations for categorical features."""
        if y is None and self.target_col in X.columns:
            y = X[self.target_col]
        
        if y is None:
            logger.warning("WoEIVTransformer: No target variable provided, skipping WoE fit")
            return self
        
        # If categorical_cols not specified, infer from X
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in self.categorical_cols:
            if col not in X.columns:
                continue
            
            try:
                woe_dict, iv_score = self._compute_woe(X[col], y)
                self.woe_dicts_[col] = woe_dict
                self.iv_scores_[col] = iv_score
                logger.debug(f"WoE fitted for '{col}': IV = {iv_score:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute WoE for '{col}': {str(e)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using WoE."""
        X_copy = X.copy()
        
        for col, woe_dict in self.woe_dicts_.items():
            if col in X_copy.columns:
                # Create WoE-encoded version
                X_copy[f'{col}_woe'] = X_copy[col].map(woe_dict).fillna(0)
        
        logger.debug(f"Applied WoE transformation to {len(self.woe_dicts_)} columns")
        return X_copy
    
    @staticmethod
    def _compute_woe(feature: pd.Series, target: pd.Series) -> Tuple[Dict, float]:
        """
        Compute WoE dictionary and IV score for a single feature.
        
        Returns:
            Tuple of (woe_dict, iv_score)
        """
        # Create contingency table
        contingency = pd.crosstab(feature, target)
        
        if contingency.shape[1] != 2:
            raise ValueError("Target must be binary (0/1)")
        
        # Get event/non-event counts
        n_events = contingency.iloc[:, 1].sum()
        n_non_events = contingency.iloc[:, 0].sum()
        
        woe_dict = {}
        iv_score = 0
        
        for category in contingency.index:
            events = contingency.loc[category, 1]
            non_events = contingency.loc[category, 0]
            
            # Calculate percentages
            pct_events = events / n_events if n_events > 0 else 0
            pct_non_events = non_events / n_non_events if n_non_events > 0 else 0
            
            # Avoid log(0) by adding small epsilon
            epsilon = 1e-10
            pct_events = np.clip(pct_events, epsilon, 1 - epsilon)
            pct_non_events = np.clip(pct_non_events, epsilon, 1 - epsilon)
            
            # Calculate WoE
            woe = np.log(pct_events / pct_non_events)
            woe_dict[category] = woe
            
            # Calculate IV contribution
            iv_contribution = (pct_events - pct_non_events) * woe
            iv_score += iv_contribution
        
        return woe_dict, iv_score


# ============================================================================
# Pipeline Assembly and Utility Functions
# ============================================================================

def create_preprocessing_pipeline(
    include_woe: bool = True,
    include_log_transform: bool = True,
    numerical_scaler: str = 'standard'
) -> Pipeline:
    """
    Creates a complete preprocessing pipeline for credit risk modeling.
    
    This pipeline handles:
    1. Aggregate feature creation from transaction data (includes temporal extraction)
    2. Categorical binning
    3. Missing value imputation
    4. Outlier handling
    5. Log transformation (optional)
    6. Categorical encoding and WoE/IV (optional)
    7. Numerical scaling/normalization
    
    Args:
        include_woe: Whether to include WoE/IV transformations
        include_log_transform: Whether to apply log transformation
        numerical_scaler: Scaler type ('standard' or 'minmax')
    
    Returns:
        Fitted sklearn Pipeline object
    
    Example:
        >>> pipeline = create_preprocessing_pipeline()
        >>> df_processed = pipeline.fit_transform(df_raw)
    """
    
    steps = [
        ('aggregate_features', AggregateFeatureTransformer()),
        ('binning', CategoricalBinningTransformer()),
        ('missing_values', MissingValueHandler()),
        ('outlier_removal', OutlierRemovalTransformer()),
    ]
    
    if include_log_transform:
        steps.append(('log_transform', LogTransformTransformer()))
    
    if include_woe:
        steps.append(('woe_iv', WoEIVTransformer()))
    
    # Add scaling for numeric columns
    steps.append(('numeric_scaling', NumericalScalingTransformer(scaler_type=numerical_scaler)))
    
    pipeline = Pipeline(steps=steps)
    logger.info(f"Created preprocessing pipeline with {len(steps)} steps")
    
    return pipeline


def fit_preprocessing_pipeline(df: pd.DataFrame, 
                               target_col: str = 'FraudResult',
                               include_woe: bool = True,
                               include_log_transform: bool = True) -> Pipeline:
    """
    Fit a preprocessing pipeline on training data.
    
    Args:
        df: Training DataFrame with all columns
        target_col: Name of target column for WoE computation
        include_woe: Whether to include WoE/IV features
        include_log_transform: Whether to include log transformation
    
    Returns:
        Fitted Pipeline object
    
    Example:
        >>> pipeline = fit_preprocessing_pipeline(df_train)
        >>> df_test_processed = pipeline.transform(df_test)
    """
    validate_dataframe_structure(df)
    
    logger.info(f"Fitting preprocessing pipeline on {len(df)} records...")
    
    pipeline = create_preprocessing_pipeline(
        include_woe=include_woe,
        include_log_transform=include_log_transform
    )
    
    # Fit the pipeline
    pipeline.fit(df, y=df[target_col] if target_col in df.columns else None)
    
    logger.info("✓ Preprocessing pipeline fitted successfully")
    return pipeline


def process_data_to_model_ready(df: pd.DataFrame,
                                pipeline: Optional[Pipeline] = None,
                                fit_on_data: bool = False,
                                target_col: str = 'FraudResult') -> Tuple[pd.DataFrame, Optional[Pipeline]]:
    """
    Transform raw data into model-ready features using the preprocessing pipeline.
    
    Args:
        df: Input DataFrame with raw features
        pipeline: Fitted Pipeline object (optional). If None and fit_on_data=False,
                 a new pipeline will be created and fitted
        fit_on_data: If True, fit the pipeline on the provided data before transforming
        target_col: Name of target column for WoE computation
    
    Returns:
        Tuple of (processed_df, fitted_pipeline)
    
    Example:
        >>> df_train_ready, pipeline = process_data_to_model_ready(
        ...     df_train, fit_on_data=True
        ... )
        >>> df_test_ready, _ = process_data_to_model_ready(
        ...     df_test, pipeline=pipeline
        ... )
    """
    validate_dataframe_structure(df)
    
    if pipeline is None and not fit_on_data:
        logger.warning("No pipeline provided and fit_on_data=False. Creating new pipeline...")
        fit_on_data = True
    
    if fit_on_data:
        pipeline = fit_preprocessing_pipeline(df, target_col=target_col)
    
    logger.info(f"Processing {len(df)} records through pipeline...")
    df_processed = pipeline.transform(df)
    
    logger.info(f"✓ Processed data shape: {df_processed.shape}")
    return df_processed, pipeline


def get_feature_importance_woe(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract WoE/IV scores from a fitted preprocessing pipeline.
    
    Args:
        pipeline: Fitted preprocessing pipeline with WoE transformer
    
    Returns:
        DataFrame with feature importance scores (IV values)
    
    Example:
        >>> importance_df = get_feature_importance_woe(pipeline)
        >>> print(importance_df.sort_values('IV', ascending=False))
    """
    try:
        woe_transformer = pipeline.named_steps.get('woe_iv')
        if woe_transformer is None:
            logger.warning("Pipeline does not contain WoE transformer")
            return pd.DataFrame()
        
        iv_scores = woe_transformer.iv_scores_
        importance_df = pd.DataFrame(
            list(iv_scores.items()),
            columns=['Feature', 'IV']
        ).sort_values('IV', ascending=False)
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Failed to extract WoE importance: {str(e)}")
        return pd.DataFrame()


def get_numerical_features(df: pd.DataFrame) -> List[str]:
    """Get list of numerical columns from DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_features(df: pd.DataFrame) -> List[str]:
    """Get list of categorical columns from DataFrame."""
    return df.select_dtypes(include=['object']).columns.tolist()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """Example usage of the preprocessing pipeline."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("Credit Risk Model - Data Processing Pipeline")
    logger.info("=" * 80)
    
    # Load raw data
    df_raw = load_raw_data(validate=True)
    logger.info(f"Loaded {len(df_raw)} transactions")
    
    # Create and fit preprocessing pipeline
    logger.info("\n" + "=" * 80)
    logger.info("Creating and fitting preprocessing pipeline...")
    logger.info("=" * 80)
    
    df_processed, pipeline = process_data_to_model_ready(
        df_raw,
        fit_on_data=True,
        target_col='FraudResult'
    )
    
    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("Processing Results")
    logger.info("=" * 80)
    logger.info(f"Original shape: {df_raw.shape}")
    logger.info(f"Processed shape: {df_processed.shape}")
    logger.info(f"\nProcessed data info:")
    logger.info(df_processed.info())
    logger.info(f"\nFirst few rows of processed data:")
    logger.info(df_processed.head())
    
    # Extract and display WoE importance
    logger.info("\n" + "=" * 80)
    logger.info("Feature Importance (WoE/IV Analysis)")
    logger.info("=" * 80)
    importance_df = get_feature_importance_woe(pipeline)
    if not importance_df.empty:
        logger.info(importance_df.to_string(index=False))
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline creation and data processing complete!")
    logger.info("=" * 80)
