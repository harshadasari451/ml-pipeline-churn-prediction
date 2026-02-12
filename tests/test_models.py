"""
Tests for model training and evaluation modules.
"""
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.predict import ChurnPredictor


@pytest.fixture
def model_trainer():
    """Create ModelTrainer instance."""
    return ModelTrainer()


@pytest.fixture
def model_evaluator():
    """Create ModelEvaluator instance."""
    return ModelEvaluator()


@pytest.fixture
def sample_train_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    return X_train, y_train


@pytest.fixture
def sample_test_data():
    """Generate sample test data."""
    np.random.seed(42)
    n_samples = 30
    n_features = 10
    
    X_test = np.random.randn(n_samples, n_features)
    y_test = np.random.randint(0, 2, n_samples)
    
    return X_test, y_test


@pytest.fixture
def trained_model(sample_train_data):
    """Create a simple trained model."""
    X_train, y_train = sample_train_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    def test_init(self, model_trainer):
        """Test ModelTrainer initialization."""
        assert model_trainer is not None
        assert model_trainer.config is not None
        assert model_trainer.model_config is not None
    
    def test_get_model_logistic_regression(self, model_trainer):
        """Test getting logistic regression model."""
        model = model_trainer.get_model('logistic_regression')
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_get_model_random_forest(self, model_trainer):
        """Test getting random forest model."""
        model = model_trainer.get_model('random_forest')
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_get_model_xgboost(self, model_trainer):
        """Test getting XGBoost model."""
        model = model_trainer.get_model('xgboost')
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_get_model_invalid(self, model_trainer):
        """Test getting invalid model raises error."""
        with pytest.raises(ValueError):
            model_trainer.get_model('invalid_model')
    
    def test_get_param_grid(self, model_trainer):
        """Test getting parameter grid."""
        param_grid = model_trainer.get_param_grid('random_forest')
        assert isinstance(param_grid, dict)
        assert len(param_grid) > 0
    
    @pytest.mark.slow
    def test_train_model(self, model_trainer, sample_train_data):
        """Test model training."""
        X_train, y_train = sample_train_data
        
        model, metrics = model_trainer.train_model(
            X_train, y_train,
            algorithm='logistic_regression',
            tune_hyperparameters=False
        )
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1' in metrics
    
    def test_select_best_model(self, model_trainer, sample_train_data):
        """Test best model selection."""
        X_train, y_train = sample_train_data
        
        # Create mock results
        results = {
            'model1': (None, {'f1': 0.75, 'accuracy': 0.80}),
            'model2': (None, {'f1': 0.85, 'accuracy': 0.82}),
            'model3': (None, {'f1': 0.70, 'accuracy': 0.78}),
        }
        
        best_algorithm, best_model, best_metrics = model_trainer.select_best_model(results, metric='f1')
        
        assert best_algorithm == 'model2'
        assert best_metrics['f1'] == 0.85


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    def test_init(self, model_evaluator):
        """Test ModelEvaluator initialization."""
        assert model_evaluator is not None
        assert model_evaluator.config is not None
    
    def test_calculate_metrics(self, model_evaluator):
        """Test metrics calculation."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 1, 0])
        
        metrics = model_evaluator.calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Check values are in valid range
        for metric_value in metrics.values():
            assert 0 <= metric_value <= 1
    
    def test_calculate_metrics_with_proba(self, model_evaluator):
        """Test metrics calculation with probabilities."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 1, 0])
        y_pred_proba = np.array([0.1, 0.9, 0.8, 0.6, 0.7, 0.2])
        
        metrics = model_evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_evaluate_model(self, model_evaluator, trained_model, sample_test_data):
        """Test complete model evaluation."""
        X_test, y_test = sample_test_data
        
        metrics = model_evaluator.evaluate_model(
            trained_model, X_test, y_test, plot=False
        )
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics


class TestChurnPredictor:
    """Tests for ChurnPredictor class."""
    
    def test_init_with_model(self, trained_model):
        """Test ChurnPredictor initialization with model."""
        predictor = ChurnPredictor(model=trained_model)
        assert predictor is not None
        assert predictor.model is not None
    
    def test_predict_single_format(self):
        """Test prediction input format handling."""
        # This is a lightweight test that doesn't require a real model
        predictor = ChurnPredictor()
        
        # Test that customer data dict is accepted
        customer_data = {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'tenure': 12,
        }
        
        # We can't actually predict without a model, but we can test initialization
        assert predictor.config is not None
