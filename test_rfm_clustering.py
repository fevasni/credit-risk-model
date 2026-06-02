"""
Comprehensive test script for RFM Metrics, K-Means Clustering, and High-Risk Labeling.

This script demonstrates the complete pipeline for:
1. Computing RFM (Recency, Frequency, Monetary) metrics
2. Clustering customers into segments using K-Means
3. Identifying and assigning high-risk customer labels
4. Integrating labels back into the main dataset for model training

Usage:
    python test_rfm_clustering.py

Author: Credit Risk Modeling Team
Date: 2026-06-02
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_raw_data
from src.feature_engineering import (
    compute_rfm_features,
    cluster_customers_kmeans,
    identify_high_risk_cluster,
    assign_high_risk_labels,
    integrate_high_risk_target
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rfm_clustering_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def test_rfm_computation():
    """Test RFM metrics calculation."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: RFM Metrics Computation")
    logger.info("="*80)

    try:
        # Load data
        df = load_raw_data(validate=False)
        logger.info(f"✓ Loaded transaction data: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Compute RFM
        rfm_df = compute_rfm_features(df)

        logger.info("\n✓ RFM computation successful!")
        logger.info(f"  Output shape: {rfm_df.shape[0]:,} customers × {rfm_df.shape[1]} features")
        logger.info(f"\nSample RFM data (first 10 customers):")
        logger.info(rfm_df.head(10).to_string())

        # Basic statistics
        logger.info("\n" + "-"*80)
        logger.info("RFM Statistics Summary")
        logger.info("-"*80)
        logger.info(rfm_df[['recency_days', 'frequency', 'monetary_total']].describe().to_string())

        return rfm_df

    except Exception as e:
        logger.error(f"✗ RFM computation failed: {str(e)}", exc_info=True)
        raise


def test_clustering(rfm_df: pd.DataFrame, n_clusters: int = 3):
    """Test K-Means clustering with feature scaling."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: K-Means Clustering with Feature Scaling")
    logger.info("="*80)

    try:
        # Perform clustering
        clustered_df, metadata = cluster_customers_kmeans(
            rfm_df,
            n_clusters=n_clusters,
            random_state=42
        )

        logger.info(f"✓ Clustering successful!")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Random state: {metadata['random_state']} (for reproducibility)")

        # Cluster distribution
        logger.info("\nCluster Distribution:")
        cluster_counts = clustered_df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            pct = (count / len(clustered_df)) * 100
            logger.info(f"  Cluster {cluster_id}: {count:,} customers ({pct:.1f}%)")

        # Compute clustering quality metrics
        logger.info("\n" + "-"*80)
        logger.info("Clustering Quality Metrics")
        logger.info("-"*80)

        # Silhouette Score (higher is better, range -1 to 1)
        rfm_features = clustered_df[['recency_days', 'frequency', 'monetary_total']].values
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_features)
        
        sil_score = silhouette_score(rfm_scaled, clustered_df['cluster'])
        logger.info(f"  Silhouette Score: {sil_score:.4f} (range: -1 to 1, higher is better)")

        # Davies-Bouldin Index (lower is better)
        db_score = davies_bouldin_score(rfm_scaled, clustered_df['cluster'])
        logger.info(f"  Davies-Bouldin Index: {db_score:.4f} (lower is better)")

        # Inertia (within-cluster sum of squares)
        logger.info(f"  Inertia (WCSS): {metadata['inertia']:.2f}")

        logger.info(f"\n✓ Cluster characteristics (sorted by Recency):")
        cluster_profiles = clustered_df.groupby('cluster').agg({
            'recency_days': ['mean', 'min', 'max'],
            'frequency': ['mean', 'min', 'max'],
            'monetary_total': ['mean', 'min', 'max']
        }).round(2)
        logger.info(cluster_profiles.to_string())

        return clustered_df, metadata

    except Exception as e:
        logger.error(f"✗ Clustering failed: {str(e)}", exc_info=True)
        raise


def test_high_risk_identification(clustered_df: pd.DataFrame):
    """Test high-risk cluster identification."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: High-Risk Cluster Identification")
    logger.info("="*80)

    try:
        high_risk_cluster_id = identify_high_risk_cluster(clustered_df)

        logger.info(f"✓ High-risk cluster identified: Cluster {high_risk_cluster_id}")

        # Detailed analysis
        logger.info("\n" + "-"*80)
        logger.info("Cluster Characteristics Comparison")
        logger.info("-"*80)

        cluster_comparison = clustered_df.groupby('cluster').agg({
            'recency_days': 'mean',
            'frequency': 'mean',
            'monetary_total': 'mean',
            'AccountId': 'count'
        }).rename(columns={'AccountId': 'n_customers'}).round(2)

        cluster_comparison['is_high_risk'] = cluster_comparison.index == high_risk_cluster_id
        logger.info(cluster_comparison.to_string())

        return high_risk_cluster_id

    except Exception as e:
        logger.error(f"✗ High-risk identification failed: {str(e)}", exc_info=True)
        raise


def test_high_risk_labeling(rfm_df: pd.DataFrame):
    """Test complete high-risk labeling pipeline."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Complete High-Risk Labeling Pipeline")
    logger.info("="*80)

    try:
        labeled_df, high_risk_cluster_id = assign_high_risk_labels(
            rfm_df,
            n_clusters=3,
            random_state=42
        )

        logger.info(f"✓ High-risk labeling completed!")
        logger.info(f"\nSample labeled data (first 10 customers):")
        logger.info(labeled_df[['AccountId', 'cluster', 'is_high_risk', 'frequency', 'monetary_total']].head(10).to_string())

        # Label distribution
        logger.info("\n" + "-"*80)
        logger.info("High-Risk Label Distribution")
        logger.info("-"*80)
        
        label_dist = labeled_df['is_high_risk'].value_counts()
        for label, count in label_dist.items():
            pct = (count / len(labeled_df)) * 100
            label_name = "High-Risk" if label == 1 else "Low-Risk"
            logger.info(f"  {label_name}: {count:,} customers ({pct:.1f}%)")

        # Cross-tabulation: Cluster vs. Label
        logger.info("\nCluster vs. High-Risk Label (Cross-tabulation):")
        crosstab = pd.crosstab(
            labeled_df['cluster'],
            labeled_df['is_high_risk'],
            margins=True
        )
        crosstab.index = ['Cluster ' + str(i) if isinstance(i, int) else i for i in crosstab.index]
        crosstab.columns = ['Low-Risk', 'High-Risk', 'Total']
        logger.info(crosstab.to_string())

        return labeled_df, high_risk_cluster_id

    except Exception as e:
        logger.error(f"✗ High-risk labeling failed: {str(e)}", exc_info=True)
        raise


def test_target_integration(df: pd.DataFrame, labeled_df: pd.DataFrame):
    """Test integration of high-risk labels into transaction data."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Target Variable Integration into Transaction Data")
    logger.info("="*80)

    try:
        df_with_target = integrate_high_risk_target(df, labeled_df)

        logger.info(f"✓ Integration successful!")
        logger.info(f"  Input transactions: {df.shape[0]:,}")
        logger.info(f"  Output transactions: {df_with_target.shape[0]:,}")
        logger.info(f"  New column added: 'is_high_risk'")

        logger.info(f"\nSample transaction data with target (first 10 rows):")
        logger.info(df_with_target[['AccountId', 'TransactionId', 'Amount', 'is_high_risk']].head(10).to_string())

        # Target distribution
        logger.info("\n" + "-"*80)
        logger.info("Target Variable Distribution in Transaction Data")
        logger.info("-"*80)
        
        target_dist = df_with_target['is_high_risk'].value_counts()
        for label, count in target_dist.items():
            pct = (count / len(df_with_target)) * 100
            label_name = "High-Risk" if label == 1 else "Low-Risk"
            logger.info(f"  {label_name} transactions: {count:,} ({pct:.1f}%)")

        # Verify data integrity
        logger.info("\n" + "-"*80)
        logger.info("Data Integrity Check")
        logger.info("-"*80)
        
        missing_target = df_with_target['is_high_risk'].isna().sum()
        logger.info(f"  Missing is_high_risk values: {missing_target}")
        logger.info(f"  Target column data type: {df_with_target['is_high_risk'].dtype}")
        logger.info(f"  Unique values: {df_with_target['is_high_risk'].unique()}")

        return df_with_target

    except Exception as e:
        logger.error(f"✗ Integration failed: {str(e)}", exc_info=True)
        raise


def save_results(rfm_df: pd.DataFrame,
                labeled_df: pd.DataFrame,
                df_with_target: pd.DataFrame):
    """Save test results to CSV files."""
    logger.info("\n" + "="*80)
    logger.info("Saving Test Results")
    logger.info("="*80)

    try:
        # Save RFM features
        rfm_output = Path("data") / "rfm_features.csv"
        rfm_df.to_csv(rfm_output, index=False)
        logger.info(f"✓ RFM features saved to: {rfm_output}")

        # Save clustered data with labels
        labeled_output = Path("data") / "customers_rfm_labeled.csv"
        labeled_df.to_csv(labeled_output, index=False)
        logger.info(f"✓ Labeled customer data saved to: {labeled_output}")

        # Save transaction data with target
        target_output = Path("data") / "transactions_with_target.csv"
        df_with_target.to_csv(target_output, index=False)
        logger.info(f"✓ Transaction data with target saved to: {target_output}")

        logger.info("\n✓ All results saved successfully!")

    except Exception as e:
        logger.error(f"✗ Failed to save results: {str(e)}", exc_info=True)
        raise


def main():
    """Execute all tests."""
    logger.info("\n" + "="*80)
    logger.info("RFM CLUSTERING & HIGH-RISK LABELING TEST SUITE")
    logger.info("="*80)

    try:
        # Load transaction data
        df = load_raw_data(validate=False)

        # Test 1: RFM Computation
        rfm_df = test_rfm_computation()

        # Test 2: Clustering
        clustered_df, metadata = test_clustering(rfm_df, n_clusters=3)

        # Test 3: High-Risk Identification
        high_risk_cluster_id = test_high_risk_identification(clustered_df)

        # Test 4: High-Risk Labeling
        labeled_df, _ = test_high_risk_labeling(rfm_df)

        # Test 5: Target Integration
        df_with_target = test_target_integration(df, labeled_df)

        # Save results
        save_results(rfm_df, labeled_df, df_with_target)

        logger.info("\n" + "="*80)
        logger.info("✓ ALL TESTS PASSED SUCCESSFULLY!")
        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("✗ TEST SUITE FAILED")
        logger.error("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
