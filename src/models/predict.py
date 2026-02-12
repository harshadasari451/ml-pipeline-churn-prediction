"""
Model prediction module for inference.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow.sklearn
import numpy as np
import pandas as pd

from src.features.feature_engineering import FeatureEngineer
from src.utils.helpers import load_config, load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChurnPredictor:
    """
    Churn prediction class for making predictions on new data.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        model_path: Optional[str] = None,
        config_path: str = "config/config.yaml"
    ):
        """
        Initialize ChurnPredictor.
        
        Args:
            model: Pre-loaded model instance
            model_path: Path to saved model
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.model = model
        self.model_path = model_path
        self.feature_engineer = FeatureEngineer(config_path)
        
        # Load feature transformers
        try:
            self.feature_engineer.load_transformers()
            logger.info("Feature transformers loaded successfully")
        except FileNotFoundError:
            logger.warning("Feature transformers not found. They must be loaded before prediction.")
        
        # Load model if not provided
        if self.model is None and self.model_path:
            self.load_model(self.model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from file.
        
        Args:
            model_path: Path to the model file
        """
        logger.info(f"Loading model from {model_path}")
        
        if model_path.startswith("models:/"):
            # Load from MLflow Model Registry
            self.model = mlflow.sklearn.load_model(model_path)
        elif model_path.endswith(".pkl"):
            # Load from pickle file
            self.model = load_pickle(model_path)
        else:
            # Try loading as MLflow model
            self.model = mlflow.sklearn.load_model(model_path)
        
        logger.info("Model loaded successfully")
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        return_proba: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions on new data.
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions array or dict with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")
        
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Preprocess features
        X = self.feature_engineer.transform(df)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Get class labels
        target_col = self.config['features']['target']
        if target_col in self.feature_engineer.label_encoders:
            le = self.feature_engineer.label_encoders[target_col]
            predictions = le.inverse_transform(predictions)
        
        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'churn_probability': probabilities[:, 1]
            }
        else:
            return predictions
    
    def predict_single(
        self,
        customer_data: Dict,
        return_proba: bool = True
    ) -> Dict[str, any]:
        """
        Make prediction for a single customer.
        
        Args:
            customer_data: Dictionary with customer features
            return_proba: Whether to return probability
            
        Returns:
            Dictionary with prediction results
        """
        result = self.predict([customer_data], return_proba=return_proba)
        
        if isinstance(result, dict):
            return {
                'prediction': result['predictions'][0],
                'churn_probability': float(result['churn_probability'][0]),
                'will_churn': result['predictions'][0] == 'Yes'
            }
        else:
            return {
                'prediction': result[0],
                'will_churn': result[0] == 'Yes'
            }
    
    def predict_batch(
        self,
        customers_data: List[Dict],
        return_proba: bool = True
    ) -> List[Dict[str, any]]:
        """
        Make predictions for multiple customers.
        
        Args:
            customers_data: List of customer data dictionaries
            return_proba: Whether to return probabilities
            
        Returns:
            List of prediction result dictionaries
        """
        result = self.predict(customers_data, return_proba=return_proba)
        
        predictions = []
        if isinstance(result, dict):
            for i in range(len(result['predictions'])):
                predictions.append({
                    'prediction': result['predictions'][i],
                    'churn_probability': float(result['churn_probability'][i]),
                    'will_churn': result['predictions'][i] == 'Yes'
                })
        else:
            for pred in result:
                predictions.append({
                    'prediction': pred,
                    'will_churn': pred == 'Yes'
                })
        
        return predictions
    
    def explain_prediction(
        self,
        customer_data: Dict,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Explain prediction with feature contributions.
        
        Args:
            customer_data: Dictionary with customer features
            feature_names: List of feature names
            
        Returns:
            Dictionary with prediction and explanation
        """
        # Make prediction
        prediction = self.predict_single(customer_data)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            if feature_names is None:
                feature_names = self.feature_engineer.feature_names
            
            importances = self.model.feature_importances_
            top_features = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            prediction['top_features'] = [
                {'feature': name, 'importance': float(imp)}
                for name, imp in top_features
            ]
        
        return prediction


def main():
    """Main function for prediction testing."""
    from src.utils.logger import setup_logging
    setup_logging()
    
    # Example customer data
    example_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.50,
        'TotalCharges': 846.00
    }
    
    logger.info("Example customer data prepared")
    logger.info("To use predictor:")
    logger.info("1. Train and save a model first")
    logger.info("2. predictor = ChurnPredictor(model_path='path/to/model')")
    logger.info("3. result = predictor.predict_single(customer_data)")


if __name__ == "__main__":
    main()
