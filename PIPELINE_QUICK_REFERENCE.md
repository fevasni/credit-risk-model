# Quick Reference: Data Processing Pipeline

## 🚀 Quick Start

### 1. Load and Process Data (One-Line)

```python
from src.data_processing import process_data_to_model_ready
from src.data_loader import load_raw_data

# Load raw data
df_raw = load_raw_data()

# Process in one shot
df_ready, pipeline = process_data_to_model_ready(df_raw, fit_on_data=True)
```

### 2. Save and Reuse Pipeline

```python
import joblib

# Save fitted pipeline
joblib.dump(pipeline, 'my_pipeline.pkl')

# Load and use on new data
pipeline = joblib.load('my_pipeline.pkl')
df_new_ready = pipeline.transform(df_new)
```

---

## 📊 Pipeline Transformations

| Step # | Transformer | Input | Output | Key Action |
|--------|-------------|-------|--------|-----------|
| 1 | AggregateFeatureTransformer | 95K transactions | 416 customers | Groups by AccountId |
| 2 | CategoricalBinningTransformer | 416 customers | 416 customers | Reduces sparse categories |
| 3 | MissingValueHandler | 416 customers | 416 customers | Fills NaN values |
| 4 | OutlierRemovalTransformer | 416 customers | 416 customers | Caps extreme values |
| 5 | LogTransformTransformer | 416 customers | 416 customers | Log-transforms Amount |
| 6 | WoEIVTransformer | 416 customers | 416 customers | Encodes categorical (optional) |
| 7 | NumericalScalingTransformer | 416 customers | 416 customers | Normalizes numeric features |

---

## 🎯 Generated Features (15 Total)

### Transaction Volume & Value
```
total_transactions        # Count of transactions
total_amount             # Sum of amounts
mean_amount              # Average amount
std_amount               # Variability in amounts
max_amount               # Largest transaction
min_amount               # Smallest transaction
```

### Risk Indicators
```
fraud_count              # Fraudulent transaction count
fraud_rate               # % of fraudulent transactions
```

### Temporal Patterns
```
transaction_hour         # Most common hour
transaction_day          # Most common day
night_transaction_ratio  # % transactions at night (22-05)
weekend_transaction_ratio # % transactions on weekends
```

### Account Activity
```
days_since_first_tx      # Account age in days
recency_days             # Days since last transaction
```

---

## 💡 Configuration Examples

### Standard Configuration (Recommended)
```python
from src.data_processing import create_preprocessing_pipeline

pipeline = create_preprocessing_pipeline(
    include_woe=True,              # Enable WoE encoding
    include_log_transform=True,    # Enable log transform
    numerical_scaler='standard'    # StandardScaler
)
pipeline.fit(df_train, y=df_train['FraudResult'])
df_train_ready = pipeline.transform(df_train)
```

### Minimal Configuration
```python
pipeline = create_preprocessing_pipeline(
    include_woe=False,             # Skip WoE
    include_log_transform=False,   # Skip log
    numerical_scaler='standard'
)
```

### MinMax Scaling Configuration
```python
pipeline = create_preprocessing_pipeline(
    include_woe=True,
    include_log_transform=True,
    numerical_scaler='minmax'      # MinMaxScaler [0,1]
)
```

---

## 🔍 Feature Importance Analysis

### Get WoE/IV Scores
```python
from src.data_processing import get_feature_importance_woe

importance_df = get_feature_importance_woe(pipeline)
print(importance_df.sort_values('IV', ascending=False))

# Top features by predictive power
print(importance_df.head(10))
```

### IV Interpretation
```
IV < 0.02      → Very weak (negligible)
0.02 - 0.1     → Weak
0.1 - 0.3      → Medium
> 0.3          → Strong predictor
```

---

## 🛠️ Common Tasks

### Task 1: Full Pipeline from Scratch
```python
from src.data_loader import load_raw_data
from src.data_processing import fit_preprocessing_pipeline

# Load data
df_raw = load_raw_data(validate=True)

# Fit pipeline
pipeline = fit_preprocessing_pipeline(
    df_raw,
    target_col='FraudResult',
    include_woe=True
)

# Transform
df_train = pipeline.transform(df_raw)
print(df_train.info())
```

### Task 2: Train/Test Split with Pipeline
```python
from sklearn.model_selection import train_test_split
import joblib

# Split raw data
df_train_raw, df_test_raw = train_test_split(
    df_raw, test_size=0.2, random_state=42
)

# Fit on training data only
pipeline = fit_preprocessing_pipeline(df_train_raw)

# Transform both sets
df_train = pipeline.transform(df_train_raw)
df_test = pipeline.transform(df_test_raw)

# Save pipeline for production
joblib.dump(pipeline, 'production_pipeline.pkl')
```

### Task 3: Production Scoring
```python
import joblib

# Load fitted pipeline
pipeline = joblib.load('production_pipeline.pkl')

# Score new transaction batch
df_new_transactions = pd.read_csv('new_transactions.csv')
df_scored = pipeline.transform(df_new_transactions)

# Use for model prediction
df_scored['risk_score'] = model.predict_proba(df_scored)[:, 1]
```

### Task 4: Custom Missing Value Strategy
```python
from src.data_processing import MissingValueHandler, create_preprocessing_pipeline

# Create custom handler
missing_handler = MissingValueHandler(
    numeric_strategy='median',  # Use median instead of mean
    categorical_strategy='most_frequent'
)

# Manual pipeline
from sklearn.pipeline import Pipeline
custom_pipeline = Pipeline([
    ('aggregate_features', AggregateFeatureTransformer()),
    ('binning', CategoricalBinningTransformer()),
    ('missing_values', missing_handler),  # Custom handler
    ('outlier_removal', OutlierRemovalTransformer()),
    ('log_transform', LogTransformTransformer()),
    ('woe_iv', WoEIVTransformer()),
    ('numeric_scaling', NumericalScalingTransformer())
])
```

### Task 5: Pipeline Inspection
```python
# View pipeline steps
print(pipeline.steps)
print(pipeline.named_steps.keys())

# Get individual transformer
agg_transformer = pipeline.named_steps['aggregate_features']
print(agg_transformer.reference_date)

# Access fitted statistics
scaler = pipeline.named_steps['numeric_scaling']
print(f"Mean: {scaler.scaler_.mean_}")
print(f"Std: {scaler.scaler_.scale_}")
```

---

## ⚠️ Common Pitfalls

### ❌ Don't: Fit pipeline on test data
```python
# WRONG!
pipeline.fit(df_test)
df_test = pipeline.transform(df_test)
```

### ✅ Do: Fit on train, transform both
```python
# CORRECT
pipeline.fit(df_train)
df_train = pipeline.transform(df_train)
df_test = pipeline.transform(df_test)
```

### ❌ Don't: Use raw data directly in model
```python
# WRONG!
model.fit(df_raw, y)
```

### ✅ Do: Use processed pipeline output
```python
# CORRECT
df_ready, pipeline = process_data_to_model_ready(df_raw, fit_on_data=True)
model.fit(df_ready, y)
```

---

## 📈 Performance Benchmarks

| Operation | Time | Data Size |
|-----------|------|-----------|
| Load 95K transactions | 0.5 sec | 61 MB |
| Fit pipeline | 0.3 sec | 1K sample |
| Transform 1K records | 0.15 sec | → 416 customers |
| Transform 95K records | ~1.5 sec | → ~40K customers |
| Scaling 14 numeric features | 0.1 sec | ~400 records |

---

## 🐛 Debugging Tips

### Print Pipeline Steps
```python
for name, transformer in pipeline.steps:
    print(f"{name}: {transformer}")
```

### Check Feature Names After Each Step
```python
# After step N
Xt = pipeline[:(N+1)].transform(X)
print(Xt.columns)
print(Xt.dtypes)
print(Xt.isna().sum())
```

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validate Output
```python
# Check for NaN
assert df_ready.isna().sum().sum() == 0, "Found NaN values!"

# Check for inf
import numpy as np
assert not np.isinf(df_ready.values).any(), "Found inf values!"

# Check numerical scale
print(df_ready.describe())
# Mean should be ~0, std ~1 for StandardScaler
```

---

## 📚 See Also

- **Full Documentation**: `DATA_PROCESSING_GUIDE.md`
- **Implementation Details**: `FEATURE_ENGINEERING_SUMMARY.md`
- **Code**: `src/data_processing.py`
- **Test**: `test_data_processing.py`

---

## ✅ Checklist: Before Training Model

- [ ] Pipeline fitted on training data only
- [ ] Both train and test data transformed with same pipeline
- [ ] Zero missing values in processed data
- [ ] No infinite or NaN values
- [ ] Numerical features scaled (mean≈0, std≈1)
- [ ] Categorical features encoded or removed
- [ ] Feature count matches expected output (15)
- [ ] Customer-level aggregation applied (not transaction-level)
- [ ] Pipeline saved for production deployment
- [ ] WoE/IV scores extracted for feature importance

