"""
Credit Risk Modeling Pipeline

A comprehensive module for building interpretable credit risk models
aligned with Basel II regulatory requirements.

Key modules:
- constants: Configuration and project constants
- validators: Data validation and quality checks
- data_loader: Data loading with defensive checks
- feature_engineering: Feature creation and transformations
- utils: Utility functions for logging, timing, plotting

Example Usage:
    >>> from src.data_loader import load_raw_data
    >>> from src.feature_engineering import compute_rfm_features
    >>> from src.utils import setup_logging
    >>>
    >>> logger = setup_logging(level='INFO')
    >>> df = load_raw_data(validate=True)
    >>> rfm_df = compute_rfm_features(df)
"""

# Version
__version__ = "0.1.0"
__author__ = "Bati Bank Data Science Team"

# Public API exports
from src.constants import (
    PROJECT_ROOT, DATA_DIR, RAW_DATA_FILE,
    REQUIRED_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
    MULTICOLLINEARITY_THRESHOLD, VIF_THRESHOLD
)

from src.validators import (
    validate_dataframe_structure,
    detect_missing_values,
    detect_duplicates,
    detect_outliers_iqr,
    validate_class_distribution,
    run_full_validation,
    ValidationError
)

from src.data_loader import (
    load_raw_data,
    load_data_dictionary,
    get_data_quality_summary,
    split_train_test
)

from src.feature_engineering import (
    compute_rfm_features,
    apply_log_transformation,
    bin_categorical_features,
    encode_categorical_onehot,
    scale_numeric_features,
    remove_multicollinear_features,
    create_interaction_features
)

from src.utils import (
    setup_logging,
    ensure_directory_exists,
    setup_plotting_style,
    print_section_header,
    Timer,
    validate_inputs
)

__all__ = [
    # Constants
    'PROJECT_ROOT', 'DATA_DIR', 'RAW_DATA_FILE',
    'REQUIRED_COLUMNS', 'NUMERIC_COLUMNS', 'CATEGORICAL_COLUMNS',
    'MULTICOLLINEARITY_THRESHOLD', 'VIF_THRESHOLD',
    
    # Validators
    'validate_dataframe_structure', 'detect_missing_values',
    'detect_duplicates', 'detect_outliers_iqr',
    'validate_class_distribution', 'run_full_validation',
    'ValidationError',
    
    # Data Loader
    'load_raw_data', 'load_data_dictionary',
    'get_data_quality_summary', 'split_train_test',
    
    # Feature Engineering
    'compute_rfm_features', 'apply_log_transformation',
    'bin_categorical_features', 'encode_categorical_onehot',
    'scale_numeric_features', 'remove_multicollinear_features',
    'create_interaction_features',
    
    # Utils
    'setup_logging', 'ensure_directory_exists',
    'setup_plotting_style', 'print_section_header',
    'Timer', 'validate_inputs'
]
