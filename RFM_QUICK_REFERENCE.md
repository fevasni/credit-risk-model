# Quick Reference: RFM Clustering Implementation

## What Was Implemented

### Four Core Functions in `src/feature_engineering.py`

#### 1. `compute_rfm_features(df, customer_col, date_col, amount_col, reference_date)`
Calculates customer-level RFM metrics from transaction data.
```python
rfm_df = compute_rfm_features(df)
# Output: DataFrame with 3,633 customers and 8 RFM features
```

#### 2. `cluster_customers_kmeans(rfm_df, n_clusters=3, rfm_columns, random_state=42)`
Performs K-Means clustering with feature scaling.
```python
clustered_df, metadata = cluster_customers_kmeans(rfm_df)
# Clusters customers into 3 segments with reproducible results
```

#### 3. `identify_high_risk_cluster(clustered_df, cluster_col, frequency_col, monetary_col)`
Identifies which cluster represents high-risk customers (low frequency + low monetary).
```python
high_risk_id = identify_high_risk_cluster(clustered_df)
# Returns: 1 (represents least engaged customers)
```

#### 4. `assign_high_risk_labels(rfm_df, n_clusters=3, random_state=42)`
One-stop function combining clustering + high-risk identification.
```python
labeled_df, high_risk_cluster = assign_high_risk_labels(rfm_df)
# Output: DataFrame with 'is_high_risk' column (binary: 0 or 1)
```

#### 5. `integrate_high_risk_target(transaction_df, rfm_df, customer_col)`
Merges customer-level labels back into transaction-level data.
```python
df_with_target = integrate_high_risk_target(df, labeled_df)
# Output: Transaction data with 'is_high_risk' column ready for modeling
```

## Key Results

| Metric | Value |
|--------|-------|
| Unique Customers | 3,633 |
| Total Transactions | 95,662 |
| High-Risk Customers | 1,359 (37.4%) |
| High-Risk Transactions | 7,570 (7.91%) |
| Clustering Method | K-Means (k=3) |
| Random State | 42 (reproducible) |
| Silhouette Score | 0.6362 (good) |
| Davies-Bouldin Index | 0.3506 (good) |

## Cluster Profiles

**Cluster 0 (62.6%):** Engaged Customers
- Recent transactions (11.8 days)
- High frequency (25.2 transactions)
- High spending (218,703.79)

**Cluster 1 (37.4%):** High-Risk Customers ⚠️
- Dormant (60.8 days since transaction)
- Low frequency (5.6 transactions)
- Lower spending (127,504.93)

**Cluster 2 (0.0%):** Outlier
- Single extreme customer with very high frequency
- Used for outlier detection

## Output Files

```
data/
├── rfm_features.csv              # Customer-level RFM metrics
├── customers_rfm_labeled.csv      # Customer labels + clusters
└── transactions_with_target.csv   # Transaction-level target variable
```

## Complete Pipeline Example

```python
from src.data_loader import load_raw_data
from src.feature_engineering import (
    compute_rfm_features,
    assign_high_risk_labels,
    integrate_high_risk_target
)

# 1. Load data
df = load_raw_data()

# 2. Calculate RFM
rfm_df = compute_rfm_features(df)

# 3. Cluster and label (all-in-one)
labeled_df, high_risk_cluster = assign_high_risk_labels(rfm_df)

# 4. Merge back to transaction data
df_final = integrate_high_risk_target(df, labeled_df)

# 5. Ready for modeling!
print(f"Target created: is_high_risk")
print(df_final[['AccountId', 'TransactionId', 'is_high_risk']].head(10))
```

## Testing

Run the complete test suite:
```bash
python test_rfm_clustering.py
```

Tests performed:
- ✓ RFM metrics computation
- ✓ K-Means clustering with scaling
- ✓ High-risk cluster identification
- ✓ Label assignment
- ✓ Target integration

## Key Features

✅ **Reproducibility:** Fixed random_state=42
✅ **Feature Scaling:** StandardScaler ensures proper K-Means
✅ **Validation:** Silhouette score and Davies-Bouldin metrics
✅ **Robust:** Defensive checks and error handling
✅ **Production-Ready:** Comprehensive logging and documentation

## Next Steps

1. Combine RFM with transaction-level features
2. Handle imbalanced target (7.9% high-risk)
3. Train classification models
4. Evaluate and deploy

---

For detailed documentation, see [RFM_IMPLEMENTATION_SUMMARY.md](RFM_IMPLEMENTATION_SUMMARY.md)
