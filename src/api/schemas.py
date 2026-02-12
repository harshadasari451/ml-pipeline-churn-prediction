"""
Pydantic schemas for API request/response validation.
"""
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class CustomerFeatures(BaseModel):
    """Customer features for churn prediction."""
    
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen status (0/1)")
    Partner: str = Field(..., description="Has partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")
    tenure: int = Field(..., ge=0, description="Number of months with company")
    PhoneService: str = Field(..., description="Has phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Has multiple lines")
    InternetService: str = Field(..., description="Internet service type")
    OnlineSecurity: str = Field(..., description="Has online security")
    OnlineBackup: str = Field(..., description="Has online backup")
    DeviceProtection: str = Field(..., description="Has device protection")
    TechSupport: str = Field(..., description="Has tech support")
    StreamingTV: str = Field(..., description="Has streaming TV")
    StreamingMovies: str = Field(..., description="Has streaming movies")
    Contract: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Has paperless billing (Yes/No)")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges amount")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v
    
    @validator('SeniorCitizen')
    def validate_senior_citizen(cls, v):
        if v not in [0, 1]:
            raise ValueError('SeniorCitizen must be 0 or 1')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.50,
                "TotalCharges": 846.00
            }
        }


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    
    customer: CustomerFeatures = Field(..., description="Customer features")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    customers: List[CustomerFeatures] = Field(..., description="List of customer features")
    
    @validator('customers')
    def validate_customers_count(cls, v):
        if len(v) == 0:
            raise ValueError('At least one customer must be provided')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 customers allowed per batch')
        return v


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    
    prediction: str = Field(..., description="Churn prediction (Yes/No)")
    churn_probability: Optional[float] = Field(None, description="Probability of churn (0-1)")
    will_churn: bool = Field(..., description="Boolean indicator of churn prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "Yes",
                "churn_probability": 0.75,
                "will_churn": True
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    churn_count: int = Field(..., description="Number of predicted churns")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": "Yes",
                        "churn_probability": 0.75,
                        "will_churn": True
                    },
                    {
                        "prediction": "No",
                        "churn_probability": 0.25,
                        "will_churn": False
                    }
                ],
                "total_count": 2,
                "churn_count": 1
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_name: str = Field(..., description="Model name")
    model_version: Optional[str] = Field(None, description="Model version")
    features_count: Optional[int] = Field(None, description="Number of features")
    model_type: Optional[str] = Field(None, description="Type of model")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "churn_classifier",
                "model_version": "1.0.0",
                "features_count": 25,
                "model_type": "XGBClassifier"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Prediction failed",
                "detail": "Invalid input data"
            }
        }
