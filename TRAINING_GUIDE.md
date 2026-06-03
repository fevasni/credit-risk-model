# Model Training and Experiment Tracking Guide

## Overview

The complete machine learning training pipeline with hyperparameter tuning and MLflow experiment tracking is implemented in `src/train.py`. This guide explains how to run training, track experiments, and register models.

## Quick Start

### 1. Run Training Pipeline

```bash
cd "c:\Users\PC\Documents\New folder (2)\credit-risk-model"
python src/train.py
```

### 2. View MLflow Dashboard

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open `http://localhost:5000` in your browser.

---

## Features

### Data Preparation
- ✓ Load transaction data
- ✓ Compute RFM metrics for all customers
- ✓ Assign high-risk labels using K-Means clustering
- ✓ Merge target variable into transaction data
- ✓ Train/test split with stratification (80/20)
- ✓ Feature scaling (StandardScaler)

### Model Training
Three models trained and compared:

1. **Logistic Regression**
   - Hyperparameter tuning: C values [0.001, 0.01, 0.1, 1, 10]
   - Grid Search with 5-fold CV
   - Scoring metric: ROC-AUC

2. **Random Forest**
   - Hyperparameter tuning: n_estimators, max_depth, min_samples_split, etc.
   - Randomized Search with 20 iterations
   - 5-fold cross-validation

3. **XGBoost**
   - Hyperparameter tuning: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
   - Grid Search with 5-fold CV
   - Best generalization for credit risk

### Evaluation Metrics
All models evaluated on test set using:
- **Accuracy**: Ratio of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of Precision and Recall
- **ROC-AUC**: Area under the ROC curve (threshold-independent)

### Experiment Tracking with MLflow
Every training run logs:
- Model type and name
- Hyperparameters (tuned values)
- Cross-validation scores
- Test set metrics (accuracy, precision, recall, F1, ROC-AUC)
- Model artifacts (trained models)

### Model Registry
- Best model automatically identified by ROC-AUC score
- Registered in MLflow Model Registry
- Transitioned to "Production" stage
- Ready for deployment

---

## Pipeline Workflow

```
1. Data Preparation
   ├─ Load raw transactions
   ├─ Compute RFM features
   ├─ Assign high-risk labels
   ├─ Feature scaling
   └─ Train/test split

2. Model Training & Tuning
   ├─ Logistic Regression (GridSearchCV)
   ├─ Random Forest (RandomizedSearchCV)
   └─ XGBoost (GridSearchCV)

3. Evaluation
   ├─ Predict on test set
   ├─ Calculate metrics
   ├─ Log to MLflow
   └─ Compare models

4. Model Registry
   ├─ Identify best model
   ├─ Register in MLflow
   └─ Transition to Production
```

---

## Output Files

### MLflow Tracking
- Location: `./mlruns/` (local file-based backend)
- Contains: Experiment runs, metrics, parameters, model artifacts
- View with: `mlflow ui --backend-store-uri file:./mlruns`

### Results Summary
- File: `results/training_results.json`
- Contains: Best model name, all metrics, hyperparameters, feature names

### Logs
- File: Console output with detailed training progress
- Shows: Data loading, RFM computation, clustering, model training, evaluation

---

## Hyperparameter Tuning Details

### Logistic Regression (Grid Search)
```python
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],           # Regularization strength
    'penalty': ['l2'],                          # Penalty type
    'solver': ['lbfgs'],                        # Optimization algorithm
    'max_iter': [1000]                          # Max iterations
}
```
- 5 combinations × 5-fold CV = 25 evaluations

### Random Forest (Randomized Search)
```python
param_grid = {
    'n_estimators': [50, 100, 200],             # Number of trees
    'max_depth': [5, 10, 15, None],            # Tree depth
    'min_samples_split': [2, 5],               # Min samples to split
    'min_samples_leaf': [1, 2],                # Min samples in leaf
    'max_features': ['sqrt', 'log2']           # Feature selection
}
```
- 20 random combinations × 5-fold CV = 100 evaluations

### XGBoost (Grid Search)
```python
param_grid = {
    'n_estimators': [50, 100, 200],            # Number of boosting rounds
    'max_depth': [3, 5, 7],                    # Tree depth
    'learning_rate': [0.01, 0.05, 0.1],       # Learning rate
    'subsample': [0.7, 0.9],                   # Row sampling
    'colsample_bytree': [0.7, 0.9]            # Column sampling
}
```
- 3 × 3 × 3 × 2 × 2 = 108 combinations × 5-fold CV = 540 evaluations

---

## Class and Method Reference

### ModelTrainingPipeline

#### `__init__(random_state=42)`
Initialize the pipeline and set up MLflow tracking.

#### `prepare_data(test_size=0.2)`
Load data, compute RFM, assign labels, and split into train/test.

#### `train_logistic_regression()`
Train LR with Grid Search hyperparameter tuning.

#### `train_random_forest()`
Train RF with Randomized Search hyperparameter tuning.

#### `train_xgboost()`
Train XGBoost with Grid Search hyperparameter tuning.

#### `evaluate_model(model, model_name, run_id=None)`
Evaluate model on test set and log metrics to MLflow.

#### `train_all_models()`
Train all three models and evaluate each.

#### `compare_models(results)`
Compare performance and return name of best model.

#### `register_best_model(best_model_name, results)`
Register best model in MLflow Model Registry.

#### `save_results(results, best_model_name)`
Save detailed results to JSON file.

---

## Expected Results

### Sample Output
```
================================================================================
DATA PREPARATION
================================================================================
✓ Loaded 95,662 transactions × 16 columns
✓ RFM computed for 3,633 customers
✓ Labels assigned (high-risk cluster: 1)
✓ Features prepared: 5 features

Target distribution:
  Low-risk (0): 88,092 (92.1%)
  High-risk (1): 7,570 (7.91%)

✓ Data split:
  Training set: 76,529 samples
  Test set: 19,133 samples
  Features: 5

================================================================================
MODEL TRAINING AND HYPERPARAMETER TUNING
================================================================================

Training: Logistic Regression
✓ Best parameters: {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000}
  Cross-validation ROC-AUC: 0.7234

Evaluating Logistic Regression...
  Accuracy:  0.9209
  Precision: 0.3214
  Recall:    0.4128
  F1 Score:  0.3623
  ROC-AUC:   0.7412

[Similar output for Random Forest and XGBoost...]

================================================================================
MODEL COMPARISON
================================================================================

                   Model    Accuracy  Precision    Recall  F1 Score   ROC-AUC
         Logistic Regression  0.9209     0.3214    0.4128    0.3623    0.7412
            Random Forest     0.9321     0.4156    0.5234    0.4623    0.8023
                  XGBoost     0.9412     0.4892    0.5876    0.5335    0.8234

Best Model: Xgboost
ROC-AUC: 0.8234

================================================================================
MODEL REGISTRY
================================================================================
✓ Model logged to MLflow
  Model URI: runs:/abc123.../model
✓ Model registered in Model Registry
  Name: credit-risk-model
  Version: 1
  Stage: Production

================================================================================
TRAINING PIPELINE COMPLETED SUCCESSFULLY
================================================================================

MLflow Tracking URI: file:./mlruns
Experiment: credit-risk-classification
View results with: mlflow ui --backend-store-uri file:./mlruns
```

---

## Model Selection Strategy

Models are compared primarily on **ROC-AUC** because:
1. ✓ Handles class imbalance well (7.9% high-risk)
2. ✓ Threshold-independent metric
3. ✓ Business metric: captures trade-off between TPR and FPR
4. ✓ Suitable for probabilistic predictions

Secondary metrics for decision-making:
- **Recall/Sensitivity**: Don't miss high-risk customers (cost of false negatives)
- **Precision**: Minimize false high-risk predictions
- **F1 Score**: Balance precision and recall

---

## Next Steps

### 1. Model Deployment
```python
# Load model from registry
import mlflow
model = mlflow.pyfunc.load_model("models:/credit-risk-model/Production")

# Make predictions
predictions = model.predict(new_data)
```

### 2. Monitor Performance
- Track model metrics over time
- Compare new data distributions
- Monitor prediction drift

### 3. Continuous Improvement
- Retrain with new data periodically
- Test new features (more RFM dimensions)
- Experiment with ensemble methods
- A/B test new model versions

### 4. Production Integration
- API endpoint for scoring
- Batch prediction pipelines
- Real-time scoring for new transactions

---

## Troubleshooting

### Issue: "No module named mlflow"
**Solution**: `pip install mlflow`

### Issue: "ImportError: cannot import name 'XGBClassifier'"
**Solution**: `pip install xgboost`

### Issue: Model Registry not working
**Solution**: Check that `./mlruns` directory exists and is writable

### Issue: Out of memory during training
**Solution**: Reduce GridSearchCV parameter grid or use RandomizedSearchCV

### Issue: Class imbalance affecting results
**Solution**: Recall that high-risk is only 7.9% of population - this is expected

---

## References

- MLflow Documentation: https://mlflow.org/docs/
- Scikit-Learn Model Selection: https://scikit-learn.org/stable/modules/model_selection.html
- XGBoost Parameters: https://xgboost.readthedocs.io/
- Evaluation Metrics Guide: https://scikit-learn.org/stable/modules/model_evaluation.html

---

**Last Updated**: June 2, 2026
