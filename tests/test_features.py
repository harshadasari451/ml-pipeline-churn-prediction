"""
Tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np

from src.features.feature_engineering import FeatureEngineer
from src.data.data_loader import DataLoader


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer()


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    loader = DataLoader()
    return loader.generate_synthetic_data(n_samples=100)


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_init(self, feature_engineer):
        """Test FeatureEngineer initialization."""
        assert feature_engineer is not None
        assert feature_engineer.config is not None
        assert feature_engineer.label_encoders == {}
    
    def test_handle_missing_values(self, feature_engineer, sample_data):
        """Test missing value handling."""
        # Add some missing values
        df = sample_data.copy()
        df.loc[0:5, 'TotalCharges'] = np.nan
        
        result = feature_engineer.handle_missing_values(df)
        
        # Check missing values are handled
        assert result['TotalCharges'].isnull().sum() == 0
    
    def test_encode_categorical_features(self, feature_engineer, sample_data):
        """Test categorical feature encoding."""
        df = sample_data.copy()
        
        # Encode features
        result = feature_engineer.encode_categorical_features(df, fit=True)
        
        # Check that categorical columns are now numeric
        for col in feature_engineer.features_config['categorical']:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col])
        
        # Check that encoders are saved
        assert len(feature_engineer.label_encoders) > 0
    
    def test_encode_categorical_features_fit_transform(self, feature_engineer, sample_data):
        """Test fit and transform separately."""
        df1 = sample_data.iloc[:50].copy()
        df2 = sample_data.iloc[50:].copy()
        
        # Fit on first half
        result1 = feature_engineer.encode_categorical_features(df1, fit=True)
        
        # Transform on second half
        result2 = feature_engineer.encode_categorical_features(df2, fit=False)
        
        # Both should have numeric types
        for col in feature_engineer.features_config['categorical']:
            if col in result1.columns and col in result2.columns:
                assert pd.api.types.is_numeric_dtype(result1[col])
                assert pd.api.types.is_numeric_dtype(result2[col])
    
    def test_scale_numerical_features(self, feature_engineer, sample_data):
        """Test numerical feature scaling."""
        df = sample_data.copy()
        
        # Scale features
        result = feature_engineer.scale_numerical_features(df, fit=True)
        
        # Check that numerical columns are scaled
        for col in feature_engineer.features_config['numerical']:
            if col in result.columns:
                # Scaled values should have mean close to 0 and std close to 1
                assert abs(result[col].mean()) < 0.5
                assert abs(result[col].std() - 1.0) < 0.5
    
    def test_create_derived_features(self, feature_engineer, sample_data):
        """Test derived feature creation."""
        df = sample_data.copy()
        
        result = feature_engineer.create_derived_features(df)
        
        # Check new features are created
        assert 'AvgChargesPerTenure' in result.columns
        assert 'ChargesRatio' in result.columns
        assert 'TenureBin' in result.columns
        assert 'ChargesBin' in result.columns
    
    def test_prepare_features(self, feature_engineer, sample_data):
        """Test complete feature preparation."""
        df = sample_data.copy()
        
        X, y = feature_engineer.prepare_features(df, fit=True, include_target=True)
        
        # Check target is separated
        assert y is not None
        assert len(y) == len(X)
        
        # Check target is not in features
        assert 'Churn' not in X.columns
        
        # Check customer ID is removed
        assert 'customerID' not in X.columns
        
        # Check features are numeric
        assert X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]
        
        # Check feature names are saved
        assert feature_engineer.feature_names is not None
    
    def test_encode_target(self, feature_engineer, sample_data):
        """Test target encoding."""
        target = sample_data['Churn']
        
        encoded = feature_engineer.encode_target(target, fit=True)
        
        # Check encoding
        assert len(encoded) == len(target)
        assert set(encoded) == {0, 1}
        
        # Check encoder is saved
        assert 'Churn' in feature_engineer.label_encoders
    
    def test_transform(self, feature_engineer, sample_data):
        """Test transform method for inference."""
        # First fit on training data
        train_df = sample_data.iloc[:70].copy()
        X_train, y_train = feature_engineer.prepare_features(train_df, fit=True)
        
        # Then transform test data
        test_df = sample_data.iloc[70:].copy()
        X_test = feature_engineer.transform(test_df)
        
        # Check same number of features
        assert X_test.shape[1] == X_train.shape[1]
        
        # Check feature names match
        assert list(X_test.columns) == list(X_train.columns)
