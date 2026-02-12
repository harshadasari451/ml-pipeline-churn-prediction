"""
Model training module with multiple algorithms and MLflow tracking.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.utils.helpers import load_config, load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Model training class with MLflow integration.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ModelTrainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.model_config = self.config['model']
        self.mlflow_config = self.config['mlflow']
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
        logger.info(f"MLflow tracking URI: {self.mlflow_config['tracking_uri']}")
        logger.info(f"MLflow experiment: {self.mlflow_config['experiment_name']}")
    
    def get_model(self, algorithm: str) -> Any:
        """
        Get model instance based on algorithm name.
        
        Args:
            algorithm: Name of the algorithm
            
        Returns:
            Model instance
        """
        if algorithm == 'logistic_regression':
            return LogisticRegression(**self.model_config['logistic_regression'])
        elif algorithm == 'random_forest':
            # Use single values for base model
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.model_config['random_forest']['random_state'],
                class_weight=self.model_config['random_forest']['class_weight']
            )
        elif algorithm == 'xgboost':
            return XGBClassifier(
                n_estimators=100,
                random_state=self.model_config['xgboost']['random_state'],
                eval_metric=self.model_config['xgboost']['eval_metric'],
                use_label_encoder=False
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def get_param_grid(self, algorithm: str) -> Dict[str, list]:
        """
        Get hyperparameter grid for tuning.
        
        Args:
            algorithm: Name of the algorithm
            
        Returns:
            Parameter grid dictionary
        """
        if algorithm == 'logistic_regression':
            return {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif algorithm == 'random_forest':
            return {
                'n_estimators': self.model_config['random_forest']['n_estimators'],
                'max_depth': self.model_config['random_forest']['max_depth'],
                'min_samples_split': self.model_config['random_forest']['min_samples_split'],
                'min_samples_leaf': self.model_config['random_forest']['min_samples_leaf']
            }
        elif algorithm == 'xgboost':
            return {
                'n_estimators': self.model_config['xgboost']['n_estimators'],
                'max_depth': self.model_config['xgboost']['max_depth'],
                'learning_rate': self.model_config['xgboost']['learning_rate']
            }
        else:
            return {}
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        algorithm: str = 'logistic_regression',
        tune_hyperparameters: bool = False
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            algorithm: Algorithm name
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Tuple of (trained model, training metrics)
        """
        logger.info(f"Training {algorithm} model...")
        
        with mlflow.start_run(run_name=f"{algorithm}_model"):
            # Log parameters
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("tune_hyperparameters", tune_hyperparameters)
            mlflow.log_param("train_samples", len(X_train))
            
            if tune_hyperparameters:
                # Hyperparameter tuning
                logger.info("Performing hyperparameter tuning...")
                base_model = self.get_model(algorithm)
                param_grid = self.get_param_grid(algorithm)
                
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=self.model_config['tuning']['n_iter'],
                    cv=self.model_config['tuning']['cv_folds'],
                    scoring=self.model_config['tuning']['scoring'],
                    random_state=self.model_config['xgboost']['random_state'],
                    n_jobs=-1,
                    verbose=1
                )
                
                search.fit(X_train, y_train)
                model = search.best_estimator_
                
                # Log best parameters
                for param, value in search.best_params_.items():
                    mlflow.log_param(f"best_{param}", value)
                
                logger.info(f"Best parameters: {search.best_params_}")
            else:
                # Train with default parameters
                model = self.get_model(algorithm)
                model.fit(X_train, y_train)
            
            # Calculate training metrics
            from src.models.evaluate import ModelEvaluator
            evaluator = ModelEvaluator()
            
            train_metrics = evaluator.calculate_metrics(y_train, model.predict(X_train))
            
            # Log training metrics
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            
            # Validation metrics if provided
            if X_val is not None and y_val is not None:
                val_metrics = evaluator.calculate_metrics(y_val, model.predict(X_val))
                for metric_name, metric_value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", metric_value)
                logger.info(f"Validation metrics: {val_metrics}")
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=None  # Don't auto-register yet
            )
            
            logger.info(f"Training completed. Metrics: {train_metrics}")
            
            return model, train_metrics
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """
        Train all configured algorithms.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of algorithm names to (model, metrics) tuples
        """
        results = {}
        
        for algorithm in self.model_config['algorithms']:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {algorithm}")
            logger.info(f"{'='*50}")
            
            model, metrics = self.train_model(
                X_train, y_train, X_val, y_val,
                algorithm=algorithm,
                tune_hyperparameters=False  # Set to True for tuning
            )
            
            results[algorithm] = (model, metrics)
        
        return results
    
    def select_best_model(
        self,
        results: Dict[str, Tuple[Any, Dict[str, float]]],
        metric: str = 'f1'
    ) -> Tuple[str, Any, Dict[str, float]]:
        """
        Select the best model based on a metric.
        
        Args:
            results: Dictionary of training results
            metric: Metric to use for selection
            
        Returns:
            Tuple of (algorithm name, best model, best metrics)
        """
        best_algorithm = None
        best_score = -1
        best_model = None
        best_metrics = None
        
        for algorithm, (model, metrics) in results.items():
            score = metrics.get(metric, 0)
            logger.info(f"{algorithm}: {metric}={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_algorithm = algorithm
                best_model = model
                best_metrics = metrics
        
        logger.info(f"\nBest model: {best_algorithm} with {metric}={best_score:.4f}")
        return best_algorithm, best_model, best_metrics
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        stage: str = "Staging"
    ) -> None:
        """
        Register model in MLflow Model Registry.
        
        Args:
            model: Trained model
            model_name: Name for the registered model
            metrics: Model metrics
            stage: Model stage (Staging, Production, Archived)
        """
        with mlflow.start_run(run_name=f"register_{model_name}"):
            # Log model with signature
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_name
            )
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            logger.info(f"Model registered as '{model_name}' in stage '{stage}'")


def main():
    """Main function for model training."""
    from src.utils.logger import setup_logging
    setup_logging()
    
    # Load processed data
    logger.info("Loading processed data...")
    X_train, y_train = load_pickle("data/processed/train_processed.pkl")
    X_val, y_val = load_pickle("data/processed/val_processed.pkl")
    X_test, y_test = load_pickle("data/processed/test_processed.pkl")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Select best model
    best_algorithm, best_model, best_metrics = trainer.select_best_model(results)
    
    # Evaluate on test set
    from src.models.evaluate import ModelEvaluator
    evaluator = ModelEvaluator()
    
    test_metrics = evaluator.evaluate_model(best_model, X_test, y_test)
    logger.info(f"\nTest set metrics: {test_metrics}")
    
    # Register best model
    trainer.register_model(
        best_model,
        trainer.mlflow_config['model_name'],
        test_metrics
    )
    
    logger.info("\nModel training pipeline completed successfully!")


if __name__ == "__main__":
    main()
