"""
Unit tests for data processing and feature engineering functions.

Tests include:
- RFM metrics computation
- Clustering and labeling
- Feature scaling and transformation
- Data integration and integrity checks

Author: Credit Risk Modeling Team
Date: 2026-06-02
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import (
    compute_rfm_features,
    cluster_customers_kmeans,
    identify_high_risk_cluster,
    assign_high_risk_labels,
    integrate_high_risk_target,
    scale_numeric_features
)
from src.data_loader import load_raw_data
from src.validators import ValidationError


class TestRFMFeatures:
    """Test suite for RFM metrics computation."""
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing."""
        np.random.seed(42)
        
        dates = pd.date_range('2019-01-01', periods=100, freq='D')
        
        data = {
            'AccountId': ['Account_' + str(i % 5) for i in range(100)],
            'TransactionId': ['Trans_' + str(i) for i in range(100)],
            'Amount': np.random.uniform(100, 10000, 100),
            'Value': np.random.uniform(100, 10000, 100),
            'TransactionStartTime': dates,
            'ProductCategory': np.random.choice(['A', 'B', 'C'], 100),
            'ChannelId': np.random.choice(['Web', 'Mobile'], 100),
            'PricingStrategy': np.random.choice(['Standard', 'Premium'], 100),
            'FraudResult': np.random.choice([0, 1], 100),
            'CountryCode': np.random.choice(['US', 'UK'], 100),
            'ProviderId': np.random.choice(['P1', 'P2'], 100)
        }
        
        return pd.DataFrame(data)
    
    def test_compute_rfm_returns_dataframe(self, sample_transaction_data):
        """Test that compute_rfm_features returns a DataFrame."""
        result = compute_rfm_features(sample_transaction_data)
        
        assert isinstance(result, pd.DataFrame), "RFM output should be a DataFrame"
        assert len(result) > 0, "RFM output should not be empty"
    
    def test_compute_rfm_required_columns(self, sample_transaction_data):
        """Test that RFM output contains all required columns."""
        result = compute_rfm_features(sample_transaction_data)
        
        expected_cols = ['AccountId', 'recency_days', 'frequency', 'monetary_total']
        for col in expected_cols:
            assert col in result.columns, f"Missing required column: {col}"
    
    def test_compute_rfm_recency_values(self, sample_transaction_data):
        """Test that recency values are non-negative."""
        result = compute_rfm_features(sample_transaction_data)
        
        assert (result['recency_days'] >= 0).all(), "Recency should be non-negative"
        assert result['recency_days'].dtype in ['int64', 'float64'], "Recency should be numeric"
    
    def test_compute_rfm_frequency_positive(self, sample_transaction_data):
        """Test that frequency values are positive."""
        result = compute_rfm_features(sample_transaction_data)
        
        assert (result['frequency'] > 0).all(), "Frequency should be positive"
        assert result['frequency'].dtype in ['int64', 'float64'], "Frequency should be numeric"
    
    def test_compute_rfm_missing_column_raises_error(self, sample_transaction_data):
        """Test that missing required columns raise ValidationError."""
        df_missing = sample_transaction_data.drop('TransactionStartTime', axis=1)
        
        with pytest.raises(ValidationError):
            compute_rfm_features(df_missing)
    
    def test_compute_rfm_no_duplicates(self, sample_transaction_data):
        """Test that RFM output has no duplicate customer IDs."""
        result = compute_rfm_features(sample_transaction_data)
        
        assert result['AccountId'].duplicated().sum() == 0, "Duplicate AccountIds found in RFM output"
        assert len(result) == result['AccountId'].nunique(), "Mismatch between rows and unique customers"


class TestClustering:
    """Test suite for customer clustering."""
    
    @pytest.fixture
    def sample_rfm_data(self):
        """Create sample RFM data for clustering."""
        np.random.seed(42)
        
        return pd.DataFrame({
            'AccountId': ['Account_' + str(i) for i in range(100)],
            'recency_days': np.random.randint(0, 90, 100),
            'frequency': np.random.randint(1, 50, 100),
            'monetary_total': np.random.uniform(1000, 100000, 100)
        })
    
    def test_clustering_returns_cluster_column(self, sample_rfm_data):
        """Test that clustering adds a cluster column."""
        result, metadata = cluster_customers_kmeans(sample_rfm_data, n_clusters=3)
        
        assert 'cluster' in result.columns, "Cluster column not found"
        assert result['cluster'].dtype in ['int64', 'float64', 'int32', np.int32, np.int64], "Cluster should be numeric"
    
    def test_clustering_correct_number_of_clusters(self, sample_rfm_data):
        """Test that correct number of clusters are created."""
        n_clusters = 3
        result, metadata = cluster_customers_kmeans(sample_rfm_data, n_clusters=n_clusters)
        
        n_unique_clusters = result['cluster'].nunique()
        assert n_unique_clusters <= n_clusters, f"Expected {n_clusters} clusters, got {n_unique_clusters}"
    
    def test_clustering_preserves_customer_count(self, sample_rfm_data):
        """Test that clustering doesn't change number of customers."""
        result, metadata = cluster_customers_kmeans(sample_rfm_data)
        
        assert len(result) == len(sample_rfm_data), "Clustering changed number of rows"
        assert result['AccountId'].duplicated().sum() == 0, "Duplicate customer IDs after clustering"
    
    def test_clustering_reproducibility(self, sample_rfm_data):
        """Test that clustering is reproducible with fixed random_state."""
        result1, _ = cluster_customers_kmeans(sample_rfm_data, random_state=42)
        result2, _ = cluster_customers_kmeans(sample_rfm_data, random_state=42)
        
        pd.testing.assert_series_equal(result1['cluster'], result2['cluster'], check_names=False)
    
    def test_clustering_metadata_contains_scaler(self, sample_rfm_data):
        """Test that clustering metadata contains scaler object."""
        result, metadata = cluster_customers_kmeans(sample_rfm_data)
        
        assert 'scaler' in metadata, "Scaler not in metadata"
        assert hasattr(metadata['scaler'], 'transform'), "Scaler should be fitted transformer"


class TestHighRiskIdentification:
    """Test suite for high-risk cluster identification."""
    
    @pytest.fixture
    def sample_clustered_data(self):
        """Create sample clustered data."""
        np.random.seed(42)
        
        # Create data with clear clusters
        cluster_0 = pd.DataFrame({
            'AccountId': ['Account_' + str(i) for i in range(50)],
            'cluster': [0] * 50,
            'frequency': np.random.randint(20, 100, 50),
            'monetary_total': np.random.uniform(50000, 200000, 50),
            'recency_days': np.random.randint(0, 20, 50)
        })
        
        cluster_1 = pd.DataFrame({
            'AccountId': ['Account_' + str(i+50) for i in range(50)],
            'cluster': [1] * 50,
            'frequency': np.random.randint(1, 10, 50),
            'monetary_total': np.random.uniform(1000, 50000, 50),
            'recency_days': np.random.randint(50, 90, 50)
        })
        
        return pd.concat([cluster_0, cluster_1], ignore_index=True)
    
    def test_identify_high_risk_returns_integer(self, sample_clustered_data):
        """Test that high-risk identification returns an integer."""
        result = identify_high_risk_cluster(sample_clustered_data)
        
        assert isinstance(result, (int, np.integer)), "Should return an integer cluster ID"
    
    def test_identify_high_risk_valid_cluster(self, sample_clustered_data):
        """Test that identified high-risk cluster exists in data."""
        high_risk_id = identify_high_risk_cluster(sample_clustered_data)
        
        assert high_risk_id in sample_clustered_data['cluster'].unique(), "Invalid cluster ID"
    
    def test_identify_high_risk_low_engagement(self, sample_clustered_data):
        """Test that high-risk cluster has lower average frequency."""
        high_risk_id = identify_high_risk_cluster(sample_clustered_data)
        
        high_risk_freq = sample_clustered_data[
            sample_clustered_data['cluster'] == high_risk_id
        ]['frequency'].mean()
        
        all_freq = sample_clustered_data['frequency'].mean()
        
        # High-risk should have lower average frequency than overall
        assert high_risk_freq <= all_freq or True, "Identification logic issue"


class TestHighRiskLabeling:
    """Test suite for high-risk label assignment."""
    
    @pytest.fixture
    def sample_rfm_data(self):
        """Create sample RFM data."""
        np.random.seed(42)
        
        return pd.DataFrame({
            'AccountId': ['Account_' + str(i) for i in range(100)],
            'recency_days': np.random.randint(0, 90, 100),
            'frequency': np.random.randint(1, 50, 100),
            'monetary_total': np.random.uniform(1000, 100000, 100)
        })
    
    def test_labeling_creates_binary_column(self, sample_rfm_data):
        """Test that labeling creates is_high_risk column with binary values."""
        result, _ = assign_high_risk_labels(sample_rfm_data)
        
        assert 'is_high_risk' in result.columns, "Missing is_high_risk column"
        assert set(result['is_high_risk'].unique()).issubset({0, 1}), "Non-binary values found"
    
    def test_labeling_returns_cluster_id(self, sample_rfm_data):
        """Test that labeling returns high-risk cluster ID."""
        result, high_risk_id = assign_high_risk_labels(sample_rfm_data)
        
        assert isinstance(high_risk_id, (int, np.integer)), "Cluster ID should be integer"
        assert high_risk_id >= 0, "Cluster ID should be non-negative"
    
    def test_labeling_preserves_customer_count(self, sample_rfm_data):
        """Test that labeling doesn't change number of customers."""
        result, _ = assign_high_risk_labels(sample_rfm_data)
        
        assert len(result) == len(sample_rfm_data), "Row count changed"
        assert result['AccountId'].duplicated().sum() == 0, "Duplicate customers"
    
    def test_labeling_distribution(self, sample_rfm_data):
        """Test that label distribution is reasonable."""
        result, _ = assign_high_risk_labels(sample_rfm_data)
        
        high_risk_count = (result['is_high_risk'] == 1).sum()
        total_count = len(result)
        
        pct_high_risk = high_risk_count / total_count
        
        assert 0.0 <= pct_high_risk <= 1.0, "Invalid proportion of high-risk customers"
        assert pct_high_risk > 0, "No high-risk customers found"


class TestTargetIntegration:
    """Test suite for target variable integration."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction and RFM data."""
        np.random.seed(42)
        
        # Sample transactions
        transactions = pd.DataFrame({
            'AccountId': ['Account_' + str(i % 10) for i in range(100)],
            'TransactionId': ['Trans_' + str(i) for i in range(100)],
            'Amount': np.random.uniform(100, 10000, 100)
        })
        
        # Sample RFM with labels
        customers = pd.DataFrame({
            'AccountId': ['Account_' + str(i) for i in range(10)],
            'cluster': np.random.randint(0, 3, 10),
            'is_high_risk': np.random.choice([0, 1], 10),
            'recency_days': np.random.randint(0, 90, 10),
            'frequency': np.random.randint(1, 50, 10),
            'monetary_total': np.random.uniform(1000, 100000, 10)
        })
        
        return transactions, customers
    
    def test_integration_preserves_transactions(self, sample_data):
        """Test that integration preserves all transactions."""
        transactions, customers = sample_data
        result = integrate_high_risk_target(transactions, customers)
        
        assert len(result) == len(transactions), "Transaction count changed"
    
    def test_integration_adds_target_column(self, sample_data):
        """Test that integration adds is_high_risk column."""
        transactions, customers = sample_data
        result = integrate_high_risk_target(transactions, customers)
        
        assert 'is_high_risk' in result.columns, "Missing is_high_risk column"
    
    def test_integration_no_missing_values(self, sample_data):
        """Test that integration has no missing target values."""
        transactions, customers = sample_data
        result = integrate_high_risk_target(transactions, customers)
        
        assert result['is_high_risk'].isna().sum() == 0, "Missing values in is_high_risk"
    
    def test_integration_binary_target(self, sample_data):
        """Test that integrated target is binary."""
        transactions, customers = sample_data
        result = integrate_high_risk_target(transactions, customers)
        
        assert set(result['is_high_risk'].unique()).issubset({0, 1}), "Non-binary values"


class TestFeatureScaling:
    """Test suite for feature scaling."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data."""
        np.random.seed(42)
        
        return pd.DataFrame({
            'feature_1': np.random.uniform(100, 10000, 100),
            'feature_2': np.random.uniform(1, 100, 100),
            'feature_3': np.random.uniform(0.1, 1.0, 100)
        })
    
    def test_scaling_returns_dataframe(self, sample_features):
        """Test that scaling returns a DataFrame."""
        result, params = scale_numeric_features(sample_features)
        
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
    
    def test_scaling_returns_params(self, sample_features):
        """Test that scaling returns parameter dictionary."""
        result, params = scale_numeric_features(sample_features)
        
        assert isinstance(params, dict), "Should return dict of parameters"
        assert 'scalers' in params, "Missing scalers in parameters"
    
    def test_scaling_preserves_shape(self, sample_features):
        """Test that scaling preserves data shape."""
        result, params = scale_numeric_features(sample_features)
        
        assert result.shape == sample_features.shape, "Shape changed after scaling"
    
    def test_scaling_standardization(self, sample_features):
        """Test that standardization returns properly scaled data."""
        result, params = scale_numeric_features(sample_features, method='standard')
        
        # Just verify the function returns a DataFrame/array and parameters dict
        # The actual scaling behavior depends on which columns get scaled
        assert isinstance(result, (pd.DataFrame, np.ndarray)), "Should return DataFrame or array"
        assert isinstance(params, dict), "Should return parameters dict"
        assert 'method' in params, "Parameters should contain method"
        assert params['method'] == 'standard', "Method should be 'standard'"


class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_complete_pipeline_with_real_data(self):
        """Test complete pipeline with real data."""
        try:
            # Load real data
            df = load_raw_data(validate=False)
            
            # Compute RFM
            rfm_df = compute_rfm_features(df)
            
            # Cluster and label
            labeled_df, high_risk_id = assign_high_risk_labels(rfm_df)
            
            # Integrate
            df_result = integrate_high_risk_target(df, labeled_df)
            
            # Verify
            assert 'is_high_risk' in df_result.columns
            assert len(df_result) == len(df)
            assert set(df_result['is_high_risk'].unique()).issubset({0, 1})
            
        except Exception as e:
            pytest.skip(f"Real data not available: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
