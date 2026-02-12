"""
Tests for data loading and validation modules.
"""
import pytest
import pandas as pd
import numpy as np

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator


@pytest.fixture
def data_loader():
    """Create DataLoader instance."""
    return DataLoader()


@pytest.fixture
def data_validator():
    """Create DataValidator instance."""
    return DataValidator()


@pytest.fixture
def sample_data(data_loader):
    """Generate sample data for testing."""
    return data_loader.generate_synthetic_data(n_samples=100)


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_init(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader is not None
        assert data_loader.config is not None
        assert data_loader.data_config is not None
    
    def test_generate_synthetic_data(self, data_loader):
        """Test synthetic data generation."""
        df = data_loader.generate_synthetic_data(n_samples=50)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert 'customerID' in df.columns
        assert 'Churn' in df.columns
        assert 'tenure' in df.columns
        assert 'MonthlyCharges' in df.columns
    
    def test_generate_synthetic_data_churn_rate(self, data_loader):
        """Test churn rate in synthetic data is reasonable."""
        df = data_loader.generate_synthetic_data(n_samples=1000)
        churn_rate = (df['Churn'] == 'Yes').mean()
        
        # Churn rate should be between 10% and 50%
        assert 0.1 <= churn_rate <= 0.5
    
    def test_split_data(self, data_loader, sample_data):
        """Test data splitting."""
        train_df, val_df, test_df = data_loader.split_data(sample_data)
        
        # Check sizes add up
        total_size = len(train_df) + len(val_df) + len(test_df)
        assert total_size == len(sample_data)
        
        # Check proportions are approximately correct
        test_ratio = len(test_df) / len(sample_data)
        assert 0.15 <= test_ratio <= 0.25
    
    def test_split_data_stratification(self, data_loader, sample_data):
        """Test stratification in data splitting."""
        train_df, val_df, test_df = data_loader.split_data(sample_data)
        
        # Check churn rates are similar across splits
        original_churn_rate = (sample_data['Churn'] == 'Yes').mean()
        train_churn_rate = (train_df['Churn'] == 'Yes').mean()
        test_churn_rate = (test_df['Churn'] == 'Yes').mean()
        
        # Churn rates should be within 10% of each other
        assert abs(train_churn_rate - original_churn_rate) < 0.1
        assert abs(test_churn_rate - original_churn_rate) < 0.1


class TestDataValidator:
    """Tests for DataValidator class."""
    
    def test_init(self, data_validator):
        """Test DataValidator initialization."""
        assert data_validator is not None
        assert data_validator.config is not None
    
    def test_validate_schema_valid(self, data_validator, sample_data):
        """Test schema validation with valid data."""
        is_valid, errors = data_validator.validate_schema(sample_data)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_schema_missing_columns(self, data_validator, sample_data):
        """Test schema validation with missing columns."""
        df = sample_data.drop(columns=['tenure'])
        is_valid, errors = data_validator.validate_schema(df)
        
        assert not is_valid
        assert len(errors) > 0
        assert 'tenure' in str(errors[0])
    
    def test_validate_data_types(self, data_validator, sample_data):
        """Test data type validation."""
        is_valid, errors = data_validator.validate_data_types(sample_data)
        assert is_valid or len(errors) == 0  # Should pass or have no critical errors
    
    def test_validate_missing_values(self, data_validator, sample_data):
        """Test missing values validation."""
        is_valid, warnings = data_validator.validate_missing_values(sample_data)
        # With synthetic data, we should have very few missing values
        assert is_valid or len(warnings) < 3
    
    def test_validate_target_distribution(self, data_validator, sample_data):
        """Test target distribution validation."""
        is_valid, errors = data_validator.validate_target_distribution(sample_data)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_target_distribution_imbalanced(self, data_validator):
        """Test target distribution with highly imbalanced data."""
        # Create imbalanced data
        df = pd.DataFrame({
            'Churn': ['No'] * 99 + ['Yes'],
            'tenure': np.random.randint(0, 73, 100),
            'MonthlyCharges': np.random.uniform(20, 100, 100),
        })
        
        is_valid, errors = data_validator.validate_target_distribution(df, min_class_percent=0.10)
        assert not is_valid
        assert len(errors) > 0
    
    def test_validate_data(self, data_validator, sample_data):
        """Test complete data validation."""
        results = data_validator.validate_data(sample_data)
        
        assert 'is_valid' in results
        assert 'errors' in results
        assert 'warnings' in results
        assert results['is_valid']  # Should pass with synthetic data
