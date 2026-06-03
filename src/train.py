"""
Model Training and Hyperparameter Tuning Pipeline with MLflow Integration.

This module implements a complete machine learning workflow including:
- Data preparation and splitting
- Multiple model training (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter tuning using Grid Search and Random Search
- Experiment tracking with MLflow
- Comprehensive model evaluation
- Model Registry management

Author: Credit Risk Modeling Team
Date: 2026-06-02
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Any
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.data_loader import load_raw_data
from src.feature_engineering import (
    compute_rfm_features,
    assign_high_risk_labels,
    integrate_high_risk_target,
    scale_numeric_features
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "credit-risk-classification"
MODEL_REGISTRY_NAME = "credit-risk-model"


class ModelTrainingPipeline:
    """
    End-to-end machine learning pipeline for credit risk classification.
    
    Handles:
    - Data loading and preparation
    - Train/test splitting
    - Model training with hyperparameter tuning
    - MLflow experiment tracking
    - Model evaluation and comparison
    - Model registration
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the training pipeline.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        logger.info("="*80)
        logger.info("Credit Risk Classification Pipeline Initialized")
        logger.info("="*80)
    
    def prepare_data(self, test_size: float = 0.2) -> None:
        """
        Load, prepare, and split data for training.
        
        Args:
            test_size: Proportion of data to use for testing (default: 0.2)
        
        Raises:
            Exception: If data preparation fails
        """
        logger.info("\n" + "="*80)
        logger.info("DATA PREPARATION")
        logger.info("="*80)
        
        try:
            # Load raw transaction data
            logger.info("Loading raw transaction data...")
            df = load_raw_data(validate=False)
            logger.info(f"✓ Loaded {df.shape[0]:,} transactions × {df.shape[1]} columns")
            
            # Compute RFM features
            logger.info("Computing RFM features...")
            rfm_df = compute_rfm_features(df)
            logger.info(f"✓ RFM computed for {len(rfm_df):,} customers")
            
            # Assign high-risk labels
            logger.info("Assigning high-risk labels...")
            labeled_df, high_risk_cluster = assign_high_risk_labels(rfm_df)
            logger.info(f"✓ Labels assigned (high-risk cluster: {high_risk_cluster})")
            
            # Integrate target back into transaction data
            logger.info("Integrating target variable...")
            df_with_target = integrate_high_risk_target(df, labeled_df)
            logger.info(f"✓ Target integrated")
            
            # Prepare features and target
            logger.info("Preparing features...")
            
            # Select numeric features for modeling
            numeric_cols = [
                'Amount', 'Value'
            ]
            
            # Add RFM features - need to merge them
            df_model = df_with_target.merge(
                rfm_df[['AccountId', 'recency_days', 'frequency', 'monetary_total']],
                on='AccountId',
                how='left'
            )
            
            # Get available features
            available_features = [col for col in numeric_cols if col in df_model.columns]
            available_features.extend([
                'recency_days', 'frequency', 'monetary_total'
            ])
            
            # Handle missing values
            df_model[available_features] = df_model[available_features].fillna(0)
            
            X = df_model[available_features].copy()
            y = df_model['is_high_risk'].astype(int).copy()
            
            logger.info(f"✓ Features prepared: {X.shape[1]} features")
            logger.info(f"  Feature columns: {', '.join(available_features)}")
            
            # Check target distribution
            target_dist = y.value_counts()
            logger.info(f"\nTarget distribution:")
            logger.info(f"  Low-risk (0): {target_dist[0]:,} ({target_dist[0]/len(y)*100:.1f}%)")
            logger.info(f"  High-risk (1): {target_dist[1]:,} ({target_dist[1]/len(y)*100:.1f}%)")
            
            # Train-test split with stratification
            logger.info(f"\nSplitting data (test_size={test_size}, random_state={self.random_state})...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y  # Preserve class distribution
            )
            
            logger.info(f"✓ Data split:")
            logger.info(f"  Training set: {self.X_train.shape[0]:,} samples")
            logger.info(f"  Test set: {self.X_test.shape[0]:,} samples")
            logger.info(f"  Features: {self.X_train.shape[1]}")
            
            self.feature_names = available_features
            
            # Feature scaling
            logger.info("\nScaling features (StandardScaler)...")
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            logger.info(f"✓ Features scaled (mean=0, std=1)")
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}", exc_info=True)
            raise
    
    def train_logistic_regression(self) -> Dict[str, Any]:
        """
        Train Logistic Regression with hyperparameter tuning.
        
        Returns:
            Dictionary with model, best parameters, and training results
        """
        logger.info("\n" + "-"*80)
        logger.info("Training: Logistic Regression")
        logger.info("-"*80)
        
        with mlflow.start_run(run_name="logistic-regression"):
            try:
                # Hyperparameter grid
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs'],
                    'max_iter': [1000]
                }
                
                # Grid Search
                logger.info(f"Running Grid Search with {len(param_grid['C'])} C values...")
                gs = GridSearchCV(
                    LogisticRegression(random_state=self.random_state, n_jobs=-1),
                    param_grid,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                gs.fit(self.X_train, self.y_train)
                
                best_model = gs.best_estimator_
                logger.info(f"✓ Best parameters: {gs.best_params_}")
                logger.info(f"  Cross-validation ROC-AUC: {gs.best_score_:.4f}")
                
                # Log parameters
                mlflow.log_params(gs.best_params_)
                mlflow.log_param("tuning_method", "grid_search")
                
                return {
                    'model': best_model,
                    'params': gs.best_params_,
                    'cv_score': gs.best_score_,
                    'gs': gs
                }
                
            except Exception as e:
                logger.error(f"Logistic Regression training failed: {str(e)}", exc_info=True)
                raise
    
    def train_random_forest(self) -> Dict[str, Any]:
        """
        Train Random Forest with hyperparameter tuning.
        
        Returns:
            Dictionary with model, best parameters, and training results
        """
        logger.info("\n" + "-"*80)
        logger.info("Training: Random Forest")
        logger.info("-"*80)
        
        with mlflow.start_run(run_name="random-forest"):
            try:
                # Hyperparameter grid
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2']
                }
                
                # Randomized Search (more efficient than Grid Search)
                logger.info("Running Randomized Search (n_iter=20)...")
                rs = RandomizedSearchCV(
                    RandomForestClassifier(
                        random_state=self.random_state,
                        n_jobs=-1
                    ),
                    param_grid,
                    n_iter=20,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=self.random_state
                )
                rs.fit(self.X_train, self.y_train)
                
                best_model = rs.best_estimator_
                logger.info(f"✓ Best parameters: {rs.best_params_}")
                logger.info(f"  Cross-validation ROC-AUC: {rs.best_score_:.4f}")
                
                # Log parameters
                mlflow.log_params(rs.best_params_)
                mlflow.log_param("tuning_method", "randomized_search")
                
                return {
                    'model': best_model,
                    'params': rs.best_params_,
                    'cv_score': rs.best_score_,
                    'rs': rs
                }
                
            except Exception as e:
                logger.error(f"Random Forest training failed: {str(e)}", exc_info=True)
                raise
    
    def train_xgboost(self) -> Dict[str, Any]:
        """
        Train XGBoost with hyperparameter tuning.
        
        Returns:
            Dictionary with model, best parameters, and training results
        """
        logger.info("\n" + "-"*80)
        logger.info("Training: XGBoost")
        logger.info("-"*80)
        
        with mlflow.start_run(run_name="xgboost"):
            try:
                # Hyperparameter grid
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.9],
                    'colsample_bytree': [0.7, 0.9]
                }
                
                # Grid Search
                logger.info(f"Running Grid Search (n_combinations={np.prod([len(v) for v in param_grid.values()])})...")
                gs = GridSearchCV(
                    xgb.XGBClassifier(
                        random_state=self.random_state,
                        n_jobs=-1,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    ),
                    param_grid,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                gs.fit(self.X_train, self.y_train)
                
                best_model = gs.best_estimator_
                logger.info(f"✓ Best parameters: {gs.best_params_}")
                logger.info(f"  Cross-validation ROC-AUC: {gs.best_score_:.4f}")
                
                # Log parameters
                mlflow.log_params(gs.best_params_)
                mlflow.log_param("tuning_method", "grid_search")
                
                return {
                    'model': best_model,
                    'params': gs.best_params_,
                    'cv_score': gs.best_score_,
                    'gs': gs
                }
                
            except Exception as e:
                logger.error(f"XGBoost training failed: {str(e)}", exc_info=True)
                raise
    
    def evaluate_model(self, model, model_name: str, run_id: str = None) -> Dict[str, float]:
        """
        Evaluate model on test set with comprehensive metrics.
        
        Args:
            model: Trained model
            model_name: Name of the model for logging
            run_id: MLflow run ID for logging metrics
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        # Get predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        # Log metrics to MLflow
        if run_id:
            with mlflow.start_run(run_id=run_id):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
        
        # Print results
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_all_models(self) -> Dict[str, Dict]:
        """
        Train all models and compare performance.
        
        Returns:
            Dictionary with results for all models
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL TRAINING AND HYPERPARAMETER TUNING")
        logger.info("="*80)
        
        results = {}
        
        try:
            # Train Logistic Regression
            lr_result = self.train_logistic_regression()
            y_pred_lr = lr_result['model'].predict(self.X_test)
            y_pred_proba_lr = lr_result['model'].predict_proba(self.X_test)[:, 1]
            metrics_lr = self.evaluate_model(
                lr_result['model'],
                "Logistic Regression"
            )
            results['logistic_regression'] = {
                'model': lr_result['model'],
                'metrics': metrics_lr,
                'cv_score': lr_result['cv_score'],
                'params': lr_result['params'],
                'predictions': y_pred_lr,
                'probabilities': y_pred_proba_lr
            }
            mlflow.log_metrics(metrics_lr)
            
            # Train Random Forest
            rf_result = self.train_random_forest()
            y_pred_rf = rf_result['model'].predict(self.X_test)
            y_pred_proba_rf = rf_result['model'].predict_proba(self.X_test)[:, 1]
            metrics_rf = self.evaluate_model(
                rf_result['model'],
                "Random Forest"
            )
            results['random_forest'] = {
                'model': rf_result['model'],
                'metrics': metrics_rf,
                'cv_score': rf_result['cv_score'],
                'params': rf_result['params'],
                'predictions': y_pred_rf,
                'probabilities': y_pred_proba_rf
            }
            mlflow.log_metrics(metrics_rf)
            
            # Train XGBoost
            xgb_result = self.train_xgboost()
            y_pred_xgb = xgb_result['model'].predict(self.X_test)
            y_pred_proba_xgb = xgb_result['model'].predict_proba(self.X_test)[:, 1]
            metrics_xgb = self.evaluate_model(
                xgb_result['model'],
                "XGBoost"
            )
            results['xgboost'] = {
                'model': xgb_result['model'],
                'metrics': metrics_xgb,
                'cv_score': xgb_result['cv_score'],
                'params': xgb_result['params'],
                'predictions': y_pred_xgb,
                'probabilities': y_pred_proba_xgb
            }
            mlflow.log_metrics(metrics_xgb)
            
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise
    
    def compare_models(self, results: Dict[str, Dict]) -> str:
        """
        Compare model performance and identify best model.
        
        Args:
            results: Dictionary with results for all models
        
        Returns:
            Name of best model
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        
        comparison_data = []
        
        for model_name, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        logger.info("\n" + comparison_df.to_string(index=False))
        
        # Find best model by ROC-AUC
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['roc_auc'])
        best_roc_auc = results[best_model_name]['metrics']['roc_auc']
        
        logger.info(f"\nBest Model: {best_model_name.replace('_', ' ').title()}")
        logger.info(f"ROC-AUC: {best_roc_auc:.4f}")
        
        return best_model_name
    
    def register_best_model(self, best_model_name: str, results: Dict[str, Dict]) -> None:
        """
        Register best model in MLflow Model Registry.
        
        Args:
            best_model_name: Name of the best model
            results: Dictionary with all model results
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL REGISTRY")
        logger.info("="*80)
        
        try:
            best_model = results[best_model_name]['model']
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            
            # Log model to MLflow
            if best_model_name == 'xgboost':
                mlflow.xgboost.log_model(best_model, "model")
            else:
                mlflow.sklearn.log_model(best_model, "model")
            
            logger.info(f"✓ Model logged to MLflow")
            logger.info(f"  Model URI: {model_uri}")
            
            # Register in Model Registry
            try:
                model_version = mlflow.register_model(
                    model_uri,
                    MODEL_REGISTRY_NAME
                )
                logger.info(f"✓ Model registered in Model Registry")
                logger.info(f"  Name: {MODEL_REGISTRY_NAME}")
                logger.info(f"  Version: {model_version.version}")
                
                # Transition to Production
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=MODEL_REGISTRY_NAME,
                    version=model_version.version,
                    stage="Production"
                )
                logger.info(f"  Stage: Production")
                
            except Exception as e:
                logger.warning(f"Could not register model: {str(e)}")
                logger.info("Model logged but not registered (model may already exist)")
        
        except Exception as e:
            logger.error(f"Model registration failed: {str(e)}", exc_info=True)
    
    def save_results(self, results: Dict[str, Dict], best_model_name: str) -> None:
        """
        Save detailed results to file.
        
        Args:
            results: Dictionary with all model results
            best_model_name: Name of the best model
        """
        output_file = Path("results") / "training_results.json"
        output_file.parent.mkdir(exist_ok=True)
        
        # Prepare results for serialization
        results_summary = {}
        for model_name, result in results.items():
            results_summary[model_name] = {
                'metrics': result['metrics'],
                'cv_score': float(result['cv_score']),
                'params': result['params']
            }
        
        output_data = {
            'best_model': best_model_name,
            'models': results_summary,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"✓ Results saved to {output_file}")


def main():
    """Main execution function."""
    try:
        # Initialize pipeline
        pipeline = ModelTrainingPipeline(random_state=42)
        
        # Prepare data
        pipeline.prepare_data(test_size=0.2)
        
        # Train all models
        results = pipeline.train_all_models()
        
        # Compare models
        best_model_name = pipeline.compare_models(results)
        
        # Register best model
        pipeline.register_best_model(best_model_name, results)
        
        # Save results
        pipeline.save_results(results, best_model_name)
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"\nMLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        logger.info(f"Experiment: {EXPERIMENT_NAME}")
        logger.info(f"View results with: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        
        return 0
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("TRAINING PIPELINE FAILED")
        logger.error("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
