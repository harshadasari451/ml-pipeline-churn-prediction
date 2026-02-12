"""
Predictions router for churn prediction endpoints.
"""
from typing import List

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Single Prediction",
    description="Predict churn for a single customer",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    }
)
async def predict_single(request: PredictionRequest):
    """
    Predict churn for a single customer.
    
    Args:
        request: Prediction request with customer features
        
    Returns:
        PredictionResponse with churn prediction
        
    Raises:
        HTTPException: If prediction fails
    """
    from src.api.main import app
    
    try:
        # Check if predictor is loaded
        if not hasattr(app.state, 'predictor') or app.state.predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please wait for model initialization."
            )
        
        # Convert request to dict
        customer_data = request.customer.dict()
        
        # Make prediction
        result = app.state.predictor.predict_single(customer_data, return_proba=True)
        
        return PredictionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Predictions",
    description="Predict churn for multiple customers",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    }
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers.
    
    Args:
        request: Batch prediction request with multiple customer features
        
    Returns:
        BatchPredictionResponse with predictions for all customers
        
    Raises:
        HTTPException: If prediction fails
    """
    from src.api.main import app
    
    try:
        # Check if predictor is loaded
        if not hasattr(app.state, 'predictor') or app.state.predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please wait for model initialization."
            )
        
        # Convert requests to list of dicts
        customers_data = [customer.dict() for customer in request.customers]
        
        # Make predictions
        results = app.state.predictor.predict_batch(customers_data, return_proba=True)
        
        # Convert to response models
        predictions = [PredictionResponse(**result) for result in results]
        
        # Calculate statistics
        churn_count = sum(1 for pred in predictions if pred.will_churn)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            churn_count=churn_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Model Information",
    description="Get information about the loaded model",
    responses={
        404: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        ModelInfoResponse with model metadata
        
    Raises:
        HTTPException: If model is not loaded
    """
    from src.api.main import app
    
    try:
        # Check if predictor is loaded
        if not hasattr(app.state, 'predictor') or app.state.predictor is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not loaded"
            )
        
        predictor = app.state.predictor
        
        # Get model information
        model_type = type(predictor.model).__name__ if predictor.model else "Unknown"
        features_count = len(predictor.feature_engineer.feature_names) if predictor.feature_engineer.feature_names else None
        
        return ModelInfoResponse(
            model_name=predictor.config['mlflow']['model_name'],
            model_version="1.0.0",
            features_count=features_count,
            model_type=model_type
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )
