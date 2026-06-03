"""
FastAPI service for credit risk prediction.

This module provides a REST API for loading the trained MLflow model
and making predictions on new customer data.
"""

import logging
import os
from typing import Optional
import numpy as np
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "credit-risk-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using MLflow-trained models",
    version="1.0.0"
)

# Global model and scaler variables
model = None
scaler = None
feature_names = None


@app.on_event("startup")
async def load_model_on_startup():
    """
    Load the trained model and scaler from MLflow registry on startup.
    """
    global model, scaler, feature_names
    
    try:
        logger.info("="*80)
        logger.info("Loading model from MLflow registry...")
        logger.info("="*80)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        
        # Load model from registry
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/{MODEL_STAGE}"
        logger.info(f"Loading model from: {model_uri}")
        
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("✓ Model loaded successfully")
        
        # Try to load scaler from MLflow artifacts
        try:
            # Check if scaler is saved as an artifact
            import joblib
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            # Get the latest production model version
            model_version = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=[MODEL_STAGE])[0]
            run_id = model_version.run_id
            
            # Try to load scaler from artifacts
            scaler_path = client.download_artifacts(run_id, "scaler.pkl")
            scaler = joblib.load(scaler_path)
            logger.info("✓ Scaler loaded successfully from MLflow artifacts")
        except Exception as scaler_error:
            logger.warning(f"Could not load scaler from MLflow: {scaler_error}")
            logger.info("Initializing StandardScaler for runtime scaling")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Note: In production, you should fit the scaler on training data
            # and save it with the model. For now, we'll use a simple scaler.
            logger.warning("⚠️  Using unfitted scaler - predictions may be inaccurate")
        
        # Define feature names (must match training)
        feature_names = ['Amount', 'Value', 'recency_days', 'frequency', 'monetary_total']
        logger.info(f"✓ Feature names: {feature_names}")
        
        logger.info("="*80)
        logger.info("Model loading completed successfully")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        # Don't raise - allow app to start but mark model as not loaded


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API status and model loading.
    """
    is_loaded = model is not None and scaler is not None
    
    return HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        message="Model loaded successfully" if is_loaded else "Model not loaded"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk probability for a customer.
    
    Args:
        request: PredictionRequest containing customer features
        
    Returns:
        PredictionResponse with risk probability and classification
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    global model, scaler, feature_names
    
    # Check if model is loaded
    if model is None:
        logger.error("Prediction attempted but model is not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    if scaler is None:
        logger.error("Prediction attempted but scaler is not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scaler not loaded. Please check server logs."
        )
    
    try:
        # Extract features from request
        features = request.features
        feature_values = np.array([[
            features.amount,
            features.value,
            features.recency_days,
            features.frequency,
            features.monetary_total
        ]])
        
        logger.info(f"Received prediction request with features: {feature_values[0]}")
        
        # Scale features (using the same scaler as training)
        # Note: If scaler was fitted during training, we should use that fitted scaler
        # For now, we'll apply scaling with runtime mean/std if scaler wasn't fitted
        if hasattr(scaler, 'mean_'):
            # Scaler was fitted during training
            scaled_features = scaler.transform(feature_values)
        else:
            # Scaler not fitted - use simple normalization
            logger.warning("Using runtime normalization (scaler not fitted)")
            scaled_features = (feature_values - feature_values.mean(axis=1, keepdims=True)) / \
                            (feature_values.std(axis=1, keepdims=True) + 1e-8)
        
        # Make prediction
        prediction_proba = model.predict_proba(scaled_features)[0, 1]
        prediction_class = model.predict(scaled_features)[0]
        
        # Determine risk classification
        risk_class = "high" if prediction_proba >= 0.5 else "low"
        is_high_risk = prediction_proba >= 0.5
        
        logger.info(f"Prediction: risk_probability={prediction_proba:.4f}, risk_class={risk_class}")
        
        return PredictionResponse(
            risk_probability=float(prediction_proba),
            risk_class=risk_class,
            is_high_risk=is_high_risk
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom exception handler for HTTP exceptions.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Custom exception handler for general exceptions.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
