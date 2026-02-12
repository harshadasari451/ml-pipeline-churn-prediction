"""
FastAPI main application for churn prediction service.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import health, predictions
from src.models.predict import ChurnPredictor
from src.utils.helpers import load_config
from src.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Load configuration
config = load_config()
api_config = config['api']


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting up Churn Prediction API...")
    
    # Load model
    try:
        # Try to load from a saved model path
        model_path = os.getenv("MODEL_PATH", "models/best_model.pkl")
        
        if Path(model_path).exists():
            logger.info(f"Loading model from {model_path}")
            app.state.predictor = ChurnPredictor(model_path=model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}")
            logger.warning("API will start without a loaded model. Train and save a model first.")
            app.state.predictor = None
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.warning("API will start without a loaded model")
        app.state.predictor = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Churn Prediction API...")
    app.state.predictor = None


# Create FastAPI application
app = FastAPI(
    title=api_config['title'],
    description=api_config['description'],
    version=api_config['version'],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors.
    
    Args:
        request: The request that caused the error
        exc: The validation exception
        
    Returns:
        JSONResponse with error details
    """
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions.
    
    Args:
        request: The request that caused the error
        exc: The exception
        
    Returns:
        JSONResponse with error details
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests.
    
    Args:
        request: The incoming request
        call_next: The next middleware or route handler
        
    Returns:
        Response from the next handler
    """
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response


# Include routers
app.include_router(health.router)
app.include_router(predictions.router)


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment or config
    host = os.getenv("API_HOST", api_config['host'])
    port = int(os.getenv("API_PORT", api_config['port']))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=api_config.get('reload', False),
        log_level="info"
    )
