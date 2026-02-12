"""
Feature engineering module for creating and transforming features.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.helpers import load_config, load_pickle, save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for preprocessing and transforming features.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize FeatureEngineer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.features_config = self.config['features']
        self.data_config = self.config['data']
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Handle TotalCharges missing values
        if 'TotalCharges' in df.columns:
            # For customers with 0 tenure, set TotalCharges to 0
            df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
            
            # Fill remaining missing values with median
            if df['TotalCharges'].isnull().any():
                median_value = df['TotalCharges'].median()
                df['TotalCharges'].fillna(median_value, inplace=True)
                logger.info(f"Filled {df['TotalCharges'].isnull().sum()} missing TotalCharges with median")
        
        # Check for other missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Remaining missing values:\n{missing_counts[missing_counts > 0]}")
        
        return df
    
    def encode_categorical_features(
        self, 
        df: pd.DataFrame, 
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        for col in self.features_config['categorical']:
            if col not in df.columns:
                continue
            
            if fit:
                # Fit and transform
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.debug(f"Encoded '{col}' with {len(le.classes_)} classes")
            else:
                # Transform only
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    logger.warning(f"No encoder found for '{col}', skipping")
        
        return df
    
    def encode_target(self, target: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Encode target variable.
        
        Args:
            target: Target series
            fit: Whether to fit the encoder
            
        Returns:
            Encoded target array
        """
        target_col = self.features_config['target']
        
        if fit:
            le = LabelEncoder()
            encoded = le.fit_transform(target)
            self.label_encoders[target_col] = le
            logger.info(f"Target classes: {le.classes_}")
        else:
            if target_col in self.label_encoders:
                le = self.label_encoders[target_col]
                encoded = le.transform(target)
            else:
                raise ValueError(f"Target encoder for '{target_col}' not found. Fit first.")
        
        return encoded
    
    def scale_numerical_features(
        self, 
        df: pd.DataFrame, 
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame with scaled numerical features
        """
        df = df.copy()
        
        numerical_cols = [
            col for col in self.features_config['numerical'] 
            if col in df.columns
        ]
        
        if not numerical_cols:
            return df
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            logger.info(f"Scaled {len(numerical_cols)} numerical features")
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()
        
        # Average monthly charges per month of tenure
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df['AvgChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        
        # Total charges to monthly charges ratio
        if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
            df['ChargesRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 0.01)
        
        # Tenure bins
        if 'tenure' in df.columns:
            df['TenureBin'] = pd.cut(
                df['tenure'], 
                bins=[0, 12, 24, 48, 100], 
                labels=[0, 1, 2, 3],
                include_lowest=True
            ).cat.codes
        
        # Monthly charges bins
        if 'MonthlyCharges' in df.columns:
            df['ChargesBin'] = pd.cut(
                df['MonthlyCharges'],
                bins=[0, 35, 65, 90, 150],
                labels=[0, 1, 2, 3],
                include_lowest=True
            ).cat.codes
        
        logger.info("Created derived features")
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Complete feature preparation pipeline.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers
            include_target: Whether to separate target variable
            
        Returns:
            Tuple of (features DataFrame, target Series or None)
        """
        logger.info(f"Preparing features (fit={fit})...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create derived features
        df = self.create_derived_features(df)
        
        # Separate features and target
        target = None
        if include_target:
            target_col = self.features_config['target']
            if target_col in df.columns:
                target = df[target_col]
                df = df.drop(columns=[target_col])
        
        # Drop customer ID if present
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Scale numerical features
        df = self.scale_numerical_features(df, fit=fit)
        
        if fit:
            self.feature_names = df.columns.tolist()
            logger.info(f"Feature preparation complete. Total features: {len(self.feature_names)}")
        
        return df, target
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        features, _ = self.prepare_features(df, fit=False, include_target=False)
        return features
    
    def save_transformers(self, filepath: str = "data/processed/feature_transformers.pkl") -> None:
        """
        Save fitted transformers to disk.
        
        Args:
            filepath: Path to save transformers
        """
        transformers = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        save_pickle(transformers, filepath)
        logger.info(f"Saved transformers to {filepath}")
    
    def load_transformers(self, filepath: str = "data/processed/feature_transformers.pkl") -> None:
        """
        Load fitted transformers from disk.
        
        Args:
            filepath: Path to load transformers from
        """
        transformers = load_pickle(filepath)
        self.label_encoders = transformers['label_encoders']
        self.scaler = transformers['scaler']
        self.feature_names = transformers['feature_names']
        logger.info(f"Loaded transformers from {filepath}")


def main():
    """Main function for feature engineering."""
    from src.data.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    train_df = load_pickle(loader.data_config['train_path'])
    val_df = load_pickle(loader.data_config['val_path'])
    test_df = load_pickle(loader.data_config['test_path'])
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Prepare training features
    X_train, y_train = engineer.prepare_features(train_df, fit=True)
    
    # Encode target
    y_train_encoded = engineer.encode_target(y_train, fit=True)
    
    # Prepare validation features
    X_val, y_val = engineer.prepare_features(val_df, fit=False)
    y_val_encoded = engineer.encode_target(y_val, fit=False)
    
    # Prepare test features
    X_test, y_test = engineer.prepare_features(test_df, fit=False)
    y_test_encoded = engineer.encode_target(y_test, fit=False)
    
    # Save processed data
    save_pickle((X_train, y_train_encoded), "data/processed/train_processed.pkl")
    save_pickle((X_val, y_val_encoded), "data/processed/val_processed.pkl")
    save_pickle((X_test, y_test_encoded), "data/processed/test_processed.pkl")
    
    # Save transformers
    engineer.save_transformers()
    
    logger.info("Feature engineering completed successfully!")
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    main()
