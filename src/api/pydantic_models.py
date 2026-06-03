"""
Pydantic models for request and response validation in the Credit Risk API.

This module defines the data structures used for API requests and responses,
ensuring type safety and validation for the credit risk prediction endpoint.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np


class CustomerFeatures(BaseModel):
    """
    Input features for credit risk prediction.
    
    This model represents the customer data required for prediction,
    matching the features used during model training.
    """
    amount: float = Field(..., ge=0, description="Transaction amount")
    value: float = Field(..., ge=0, description="Transaction value")
    recency_days: float = Field(..., ge=0, description="Days since last transaction")
    frequency: float = Field(..., ge=0, description="Total transaction frequency")
    monetary_total: float = Field(..., ge=0, description="Total monetary value")

    class Config:
        schema_extra = {
            "example": {
                "amount": 1000.0,
                "value": 950.0,
                "recency_days": 15.0,
                "frequency": 25.0,
                "monetary_total": 25000.0
            }
        }

    @validator('amount', 'value', 'recency_days', 'frequency', 'monetary_total')
    def validate_non_negative(cls, v):
        """Ensure all numeric values are non-negative."""
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v


class PredictionRequest(BaseModel):
    """
    Request model for the /predict endpoint.
    
    Accepts customer features and returns risk probability.
    """
    features: CustomerFeatures

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "amount": 1000.0,
                    "value": 950.0,
                    "recency_days": 15.0,
                    "frequency": 25.0,
                    "monetary_total": 25000.0
                }
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for the /predict endpoint.
    
    Returns the predicted risk probability and classification.
    """
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of high risk (0-1)")
    risk_class: str = Field(..., description="Risk classification: 'high' or 'low'")
    is_high_risk: bool = Field(..., description="Boolean indicating if customer is high risk")

    class Config:
        schema_extra = {
            "example": {
                "risk_probability": 0.75,
                "risk_class": "high",
                "is_high_risk": True
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for the /health endpoint.
    """
    status: str
    model_loaded: bool
    message: str


class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    """
    error: str
    detail: Optional[str] = None
