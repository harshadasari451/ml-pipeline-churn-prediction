"""
Health check router for API monitoring.
"""
from fastapi import APIRouter, status

from src.api.schemas import HealthResponse
from src.utils.helpers import load_config

router = APIRouter(prefix="", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check if the API service is healthy and the model is loaded"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status
    """
    from src.api.main import app
    
    config = load_config()
    
    # Check if model is loaded
    model_loaded = hasattr(app.state, 'predictor') and app.state.predictor is not None
    
    return HealthResponse(
        status="healthy",
        version=config['api']['version'],
        model_loaded=model_loaded
    )


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Root Endpoint",
    description="Welcome message and API information"
)
async def root():
    """
    Root endpoint with welcome message.
    
    Returns:
        Dictionary with welcome message
    """
    config = load_config()
    
    return {
        "message": "Welcome to Churn Prediction API",
        "version": config['api']['version'],
        "description": config['api']['description'],
        "docs_url": "/docs",
        "health_check": "/health"
    }
