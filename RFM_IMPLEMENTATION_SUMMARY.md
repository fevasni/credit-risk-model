# RFM Metrics, K-Means Clustering & High-Risk Customer Labeling - Implementation Summary

## Overview

Successfully implemented a complete pipeline for customer risk segmentation using RFM (Recency, Frequency, Monetary) metrics, K-Means clustering, and high-risk customer identification. This pipeline creates a binary target variable (`is_high_risk`) for credit risk modeling.

## Implementation Details

### 1. RFM Metrics Calculation (`compute_rfm_features`)

**Location:** [src/feature_engineering.py](src/feature_engineering.py#L32-L116)

**Features Computed:**
- **Recency:** Days since last transaction (lower = more recent engagement)
- **Frequency:** Total transaction count per customer
- **Monetary:** Sum, mean, max, and std of transaction amounts

**Key Parameters:**
- Snapshot date: Reference date for recency calculation (default: max date in data = 2019-02-13)
- Customer grouping: AccountId
- Output: Customer-level aggregated DataFrame with 8 features

**Results from Test:**
- 3,633 unique customers identified
- Recency range: 0-90 days
- Frequency range: 1-30,893 transactions
- Monetary range: -112.5M to 83.5M

### 2. K-Means Clustering (`cluster_customers_kmeans`)

**Location:** [src/feature_engineering.py](src/feature_engineering.py#L119-L208)

**Process:**
1. Feature scaling using StandardScaler (critical for K-Means)
   - Mean = 0, Std = 1 (standardization)
2. K-Means clustering with 3 segments
3. Set `random_state=42` for reproducibility
4. Use `n_init=10` for multiple random initializations

**Clustering Quality Metrics:**
- **Silhouette Score:** 0.6362 (higher is better, range: -1 to 1) ✓ Good separation
- **Davies-Bouldin Index:** 0.3506 (lower is better) ✓ Well-separated clusters
- **Inertia (WCSS):** 4464.30

**Cluster Characteristics:**
| Cluster | Count | Recency (days) | Frequency | Monetary |
|---------|-------|-----------------|-----------|----------|
| 0       | 2,273 (62.6%) | 11.8 | 25.2 | 218,703.79 |
| 1       | 1,359 (37.4%) | 60.8 | 5.6 | 127,504.93 |
| 2       | 1 (0.0%) | 0.0 | 30,893 | -27.75M |

### 3. High-Risk Cluster Identification (`identify_high_risk_cluster`)

**Location:** [src/feature_engineering.py](src/feature_engineering.py#L211-L289)

**Logic:**
- Identified as **Cluster 1** (highest risk score = 0.9967)
- Characteristics: Low frequency (5.6 transactions) + Low monetary (127,504.93)
- Represents **1,359 customers (37.4%)** with least engagement

**Risk Profile:**
- Long recency (60.8 days since last transaction)
- Few transactions on average
- Lower average spending
- Clear indicator of customer disengagement/attrition risk

### 4. High-Risk Label Assignment (`assign_high_risk_labels`)

**Location:** [src/feature_engineering.py](src/feature_engineering.py#L292-L351)

**Output:**
Binary `is_high_risk` column with values:
- **1** = High-risk customer (in Cluster 1)
- **0** = Low-risk customer (in Clusters 0 or 2)

**Label Distribution:**
- High-risk customers: 1,359 (37.4%)
- Low-risk customers: 2,274 (62.6%)

### 5. Target Variable Integration (`integrate_high_risk_target`)

**Location:** [src/feature_engineering.py](src/feature_engineering.py#L354-L720)

**Process:**
- Merge customer-level labels back into transaction-level data
- Join on AccountId
- Handle missing values with defensive filling

**Transaction-Level Target Distribution:**
- High-risk transactions: 7,570 (7.91%)
- Low-risk transactions: 88,092 (92.09%)
- Total transactions: 95,662

**Data Integrity:**
- No missing values in target column
- Data type: int64 (binary: 0 or 1)
- Row count preserved: 95,662 transactions

## Output Files Generated

### 1. `rfm_features.csv`
Customer-level RFM features (3,633 rows × 8 columns)
- AccountId, recency_days, last_transaction_date
- frequency, monetary_total, monetary_mean, monetary_max, monetary_std

### 2. `customers_rfm_labeled.csv`
Customer-level data with cluster assignments and high-risk labels (3,633 rows)
- All RFM features + cluster + is_high_risk columns

### 3. `transactions_with_target.csv`
Transaction-level data with integrated target variable (95,662 rows)
- Original transaction features + is_high_risk column (binary target)

## Key Features

✓ **Reproducibility:** `random_state=42` ensures consistent clustering results
✓ **Feature Scaling:** StandardScaler applied before K-Means for proper distance calculations
✓ **Defensive Programming:** Error handling, validation checks, and missing value handling
✓ **Comprehensive Logging:** Detailed logs for each step of the pipeline
✓ **Quality Metrics:** Silhouette score and Davies-Bouldin index calculated
✓ **Data Integrity:** Merge verification and validation checks included

## Usage Example

```python
from src.feature_engineering import (
    compute_rfm_features,
    assign_high_risk_labels,
    integrate_high_risk_target
)
from src.data_loader import load_raw_data

# Load transaction data
df = load_raw_data()

# Step 1: Compute RFM metrics
rfm_df = compute_rfm_features(df)

# Step 2: Cluster and assign high-risk labels
labeled_df, high_risk_cluster_id = assign_high_risk_labels(rfm_df)

# Step 3: Merge labels back into transaction data
df_with_target = integrate_high_risk_target(df, labeled_df)

# Ready for model training!
print(f"Target variable created: is_high_risk (1={high_risk_cluster_id})")
print(f"Target distribution:")
print(df_with_target['is_high_risk'].value_counts())
```

## Testing

Comprehensive test suite in `test_rfm_clustering.py`:
- Test 1: RFM Metrics Computation ✓
- Test 2: K-Means Clustering with Feature Scaling ✓
- Test 3: High-Risk Cluster Identification ✓
- Test 4: Complete High-Risk Labeling Pipeline ✓
- Test 5: Target Variable Integration into Transaction Data ✓

**All tests passed successfully!**

Run tests with:
```bash
python test_rfm_clustering.py
```

## Algorithm Details

### K-Means Configuration
```python
KMeans(
    n_clusters=3,           # 3 customer segments
    random_state=42,        # Reproducible results
    n_init=10               # Multiple random initializations
)
```

### Feature Scaling
```python
StandardScaler()  # mean=0, std=1 standardization
```

### High-Risk Detection
Normalized risk score: `(freq_normalized + monetary_normalized).idxmin()`
- Lower combined score → Higher risk
- Identifies cluster with lowest engagement

## Model Training Considerations

1. **Imbalanced Target:** 92.09% low-risk vs 7.91% high-risk
   - Consider stratified sampling for train/test split
   - Apply SMOTE or class weights if needed

2. **Feature Engineering Pipeline:**
   - RFM features are customer-level aggregates
   - Use `merge` to attach to transaction-level features
   - Consider both customer-level and transaction-level features

3. **Reproducibility:**
   - All random seeds fixed (random_state=42)
   - Feature scaling parameters saved in metadata
   - Reference date recorded (2019-02-13)

## Files Modified/Created

### Modified:
- `src/feature_engineering.py` - Added clustering and labeling functions

### Created:
- `test_rfm_clustering.py` - Comprehensive test suite
- `data/rfm_features.csv` - RFM metrics output
- `data/customers_rfm_labeled.csv` - Customer labels
- `data/transactions_with_target.csv` - Transaction-level target variable

## Next Steps

1. **Feature Selection:** Combine RFM features with other transaction/customer features
2. **Model Development:** Train classification models (logistic regression, random forest, XGBoost)
3. **Validation:** Cross-validation and out-of-sample testing
4. **Deployment:** Create prediction pipeline for new customers
5. **Monitoring:** Track cluster drift and model performance over time

---

**Implementation Date:** June 2, 2026
**Status:** ✓ Complete & Tested
