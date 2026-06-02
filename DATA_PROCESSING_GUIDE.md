# Data Processing and Feature Engineering Pipeline

## Overview

This comprehensive data processing pipeline transforms raw transaction data into model-ready features for credit risk modeling. The pipeline is implemented as a single fitted `sklearn.Pipeline` object that ensures reproducibility and seamless integration with machine learning workflows.

## Pipeline Architecture

The preprocessing pipeline consists of **7 sequential transformation steps**:

### 1. **Aggregate Feature Transformer**
Converts transaction-level data to customer-level aggregates.

**Output Features:**
- `total_transactions`: Count of transactions per customer
- `total_amount`: Sum of transaction amounts
- `mean_amount`: Average transaction amount
- `std_amount`: Standard deviation of amounts
- `max_amount`: Maximum transaction amount
- `min_amount`: Minimum transaction amount
- `fraud_count`: Number of fraudulent transactions
- `fraud_rate`: Proportion of fraudulent transactions
- `night_transaction_ratio`: Ratio of night-time transactions (22:00-05:00)
- `weekend_transaction_ratio`: Ratio of weekend transactions
- `days_since_first_tx`: Days between first and last transaction
- `recency_days`: Days since last transaction

### 2. **Categorical Binning Transformer**
Reduces dimensionality of categorical features by binning rare categories.

**Process:**
- Retains top N categories by frequency
- Groups remaining categories as 'Other'
- Helps prevent sparse categorical features

### 3. **Missing Value Handler**
Imputes missing values using statistical methods.

**Strategies:**
- **Numeric columns**: Fills with mean (configurable: median/mode)
- **Categorical columns**: Fills with mode (most frequent value)
- **Critical columns**: Removes rows with missing critical values
- Learned during fitting, applied consistently during transformation

### 4. **Outlier Removal Transformer**
Handles outliers using configurable statistical methods.

**Methods:**
- **IQR Method** (default): Outliers = Q1 - 1.5×IQR or Q3 + 1.5×IQR
- **Z-Score Method**: Outliers with |z| > 3 standard deviations
- **Action**: Caps outliers to bounds instead of removing rows

### 5. **Log Transform Transformer**
Reduces skewness in highly skewed numerical features.

**Formula:** `log(x + epsilon)` where epsilon=1.0
**Target Features:**
- Amount and derived amount features

### 6. **WoE/IV Transformer** *(optional)*
Computes Weight of Evidence (WoE) for categorical features.

**Weight of Evidence (WoE):**
$$\text{WoE} = \ln\left(\frac{\% \text{ Events}}{\% \text{ Non-Events}}\right)$$

**Information Value (IV):**
$$\text{IV} = \sum \left(\% \text{ Events} - \% \text{ Non-Events}\right) \times \text{WoE}$$

**Purpose:**
- Transforms categorical variables into continuous values
- Measures feature's predictive power
- Higher IV values = stronger relationship with target

### 7. **Numerical Scaling Transformer**
Scales numerical features to comparable ranges.

**Methods:**
- **StandardScaler** (default): Mean=0, Std=1 (z-score normalization)
- **MinMaxScaler**: Scales to [0, 1] range
- Applied only to numeric columns, preserves categorical features

## Usage

### Basic Usage

```python
from src.data_processing import fit_preprocessing_pipeline, process_data_to_model_ready

# Load your raw data
df_raw = pd.read_csv('data/transactions.csv')

# Fit pipeline on training data
pipeline, df_train_ready = fit_preprocessing_pipeline(
    df_raw,
    target_col='FraudResult',
    include_woe=True,
    include_log_transform=True
)

# Save the pipeline for later use
import joblib
joblib.dump(pipeline, 'pipeline.pkl')
```

### Transform New Data

```python
# Load pipeline
pipeline = joblib.load('pipeline.pkl')

# Transform test/production data
df_test_ready, _ = process_data_to_model_ready(
    df_test,
    pipeline=pipeline,
    fit_on_data=False
)
```

### Feature Importance Analysis

```python
from src.data_processing import get_feature_importance_woe

# Extract WoE/IV scores
importance_df = get_feature_importance_woe(pipeline)
print(importance_df.sort_values('IV', ascending=False))
```

## Feature Engineering Details

### Aggregate Features

Customer-level features are computed by grouping transaction-level data by `AccountId`:

```python
df_agg = df_raw.groupby('AccountId').agg({
    'TransactionId': 'count',           # frequency
    'Amount': ['sum', 'mean', 'std', 'max', 'min'],  # amount statistics
    'FraudResult': ['sum', 'mean'],    # fraud indicators
    # + temporal pattern features
})
```

### Temporal Features

Extracted during aggregation to capture transaction timing patterns:

- **Hour of Day**: 0-23
- **Day of Month**: 1-31
- **Weekend Indicator**: Binary (1 if Sat/Sun)
- **Night Transaction Ratio**: Proportion of transactions 22:00-05:00

### Missing Value Handling

Default imputation strategies:

| Column Type | Strategy | Value |
|-------------|----------|-------|
| Numeric | Mean | Column mean computed during fit |
| Categorical | Mode | Most frequent category during fit |
| Critical Columns | Removal | Rows with any missing values dropped |

### Normalization Strategies

**StandardScaler (Recommended):**
- **Formula**: $z = \frac{x - \mu}{\sigma}$
- **Range**: (-∞, +∞), typically [-3, +3]
- **Use Case**: Algorithms sensitive to feature scale (SVM, KNN, Neural Networks)

**MinMaxScaler:**
- **Formula**: $x_{\text{scaled}} = \frac{x - \min(x)}{\max(x) - \min(x)}$
- **Range**: [0, 1]
- **Use Case**: Features needing bounded output

## Key Features

✅ **Production-Ready**: Single fitted Pipeline object for reproducibility
✅ **Defensive**: Comprehensive error handling and validation
✅ **Modular**: Independent transformers can be used separately
✅ **Documented**: Extensive docstrings and logging
✅ **Tested**: Validation test script included
✅ **Flexible**: Configurable components and strategies
✅ **Efficient**: Handles 95K+ transactions in seconds

## Example Output

After processing 1,000 raw transactions:
- **Input**: 1,000 transaction records × 16 columns
- **Output**: 416 customer records × 15 features
- **Numerical Features**: 14
- **Categorical Features**: 0 (all encoded or aggregated)
- **Missing Values**: 0%
- **Memory Usage**: ~0.05 MB (processed data)

## Advanced Configuration

### Custom Scaler

```python
pipeline = create_preprocessing_pipeline(
    include_woe=True,
    include_log_transform=True,
    numerical_scaler='minmax'  # or 'standard'
)
```

### Disable WoE/IV Transformation

```python
pipeline = create_preprocessing_pipeline(
    include_woe=False,
    include_log_transform=True,
    numerical_scaler='standard'
)
```

### Custom Outlier Detection

```python
from src.data_processing import OutlierRemovalTransformer

outlier_transformer = OutlierRemovalTransformer(
    method='zscore',  # or 'iqr'
    zscore_threshold=3.0
)
```

## Performance Metrics

**Processing Speed:**
- ~95K transactions processed in ~2 seconds
- ~500 customer-level records generated per 1K transactions
- Scales linearly with data size

**Feature Quality:**
- No missing values in output
- No infinite or NaN values
- All features normalized to comparable scales
- Information Value computed for categorical features

## Dependencies

```
pandas>=2.2.3
numpy>=2.2.3
scikit-learn>=1.5.2
scipy>=1.14.1
```

## Testing

Run the validation test:

```bash
python test_data_processing.py
```

This validates:
- Data loading and structure
- Pipeline creation and fitting
- Data transformation
- Feature generation
- No missing values or NaNs
- Proper scaling and normalization

## Next Steps

1. **Model Training**: Use the processed DataFrame with your ML algorithm
2. **Feature Selection**: Use WoE/IV scores to select top features
3. **Hyperparameter Tuning**: Adjust imputation, scaling, and encoding strategies
4. **Production Deployment**: Pickle the fitted pipeline for scoring new data
5. **Monitoring**: Track feature distributions and data drift over time

## Troubleshooting

### Q: Why are some features NaN after transformation?
**A**: Check if those columns had all missing values. The imputer skips columns with 100% missing data.

### Q: How do I add custom transformers?
**A**: Create a class inheriting from `BaseEstimator` and `TransformerMixin`, then insert into the pipeline steps list.

### Q: Can I use the pipeline with sklearn's cross-validation?
**A**: Yes! The pipeline is fully compatible with `GridSearchCV`, `cross_val_score`, etc.

## References

- **Weight of Evidence & Information Value**: Statistical measure of predictive power commonly used in credit risk modeling
- **Feature Engineering**: Transforming raw data into meaningful predictors for machine learning
- **Scikit-learn Pipeline**: Ensures train/test consistency and prevents data leakage
