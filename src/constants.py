"""
Constants and configuration for credit risk modeling pipeline.

This module centralizes all configuration values, file paths, and domain
constraints to ensure consistency across the project and facilitate maintenance.
"""

from pathlib import Path
from typing import Dict, Set

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "Scripts"

# Raw and processed data files
RAW_DATA_FILE = DATA_DIR / "data.csv"
DATA_DICTIONARY_FILE = DATA_DIR / "Xente_Variable_Definitions.csv"

# ============================================================================
# Data Validation Constants
# ============================================================================

# Expected data shape and columns
REQUIRED_COLUMNS: Set[str] = {
    "TransactionId", "AccountId", "Amount", "Value",
    "ProductCategory", "ChannelId", "PricingStrategy",
    "FraudResult", "TransactionStartTime", "CountryCode", "ProviderId"
}

# Expected data types for validation
EXPECTED_DTYPES: Dict[str, str] = {
    "TransactionId": "object",
    "AccountId": "object",
    "Amount": "float64",
    "Value": "float64",
    "ProductCategory": "object",
    "ChannelId": "object",
    "PricingStrategy": "object",
    "FraudResult": "int64",
    "TransactionStartTime": "object",  # Will be converted to datetime
    "CountryCode": "object",
    "ProviderId": "object"
}

# Numeric columns for statistical analysis
NUMERIC_COLUMNS: Set[str] = {"Amount", "Value", "FraudResult"}

# Categorical columns
CATEGORICAL_COLUMNS: Set[str] = {
    "ProductCategory", "ChannelId", "PricingStrategy",
    "CountryCode", "ProviderId"
}

# ============================================================================
# Business Rules & Thresholds
# ============================================================================

# Multicollinearity threshold for feature retention
MULTICOLLINEARITY_THRESHOLD = 0.7  # Correlation coefficient threshold
VIF_THRESHOLD = 5.0  # Variance Inflation Factor threshold for feature selection

# Outlier detection parameters
IQR_MULTIPLIER = 1.5  # For IQR method: outliers = Q1 - 1.5*IQR or Q3 + 1.5*IQR
ZSCORE_THRESHOLD = 3.0  # For Z-score method: |z| > 3 is considered outlier

# RFM Aggregation parameters
RFM_RECENCY_WINDOWS_DAYS = [30, 60, 90]  # Time windows for recency calculation
RFM_AGGREGATION_LEVEL = "AccountId"  # Primary grouping key for customer aggregation

# Fraud class imbalance handling
FRAUD_PREVALENCE_RATE = 0.002  # ~0.20% - used for stratified sampling
MIN_SAMPLES_PER_CLASS = 30  # Minimum samples required for train/test split

# Log transformation parameters
LOG_TRANSFORM_COLUMNS = ["Amount"]  # Columns requiring log transformation
LOG_EPSILON = 1.0  # Offset for log(x + epsilon) to handle zeros

# ============================================================================
# Feature Engineering Constraints
# ============================================================================

# Categorical binning rules (preserve top N categories, bin rest as "Other")
TOP_CATEGORIES_TO_KEEP = {
    "ProductCategory": 5,
    "ChannelId": 3,
    "ProviderId": 10,
    "PricingStrategy": 4
}

# Categories to exclude or validate
VALID_CHANNELS = {"Web", "Mobile", "USSD"}
VALID_PRICING_STRATEGIES = {"Standard", "Premium", "VIP"}

# ============================================================================
# Model Configuration
# ============================================================================

# Logistic Regression hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "penalty": "l2",  # L2 regularization for better generalization
    "solver": "lbfgs",  # Suitable for small datasets
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": "balanced"  # Address class imbalance
}

# Train/test split ratio
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_STATE_SEED = 42

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = "INFO"  # Can be set to DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
