"""
Data loader module for ingesting and generating churn prediction data.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split

from src.utils.helpers import ensure_dir, load_config, save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Data loader class for loading or generating churn prediction data.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataLoader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_config = self.config['data']
        self.features_config = self.config['features']
        self.fake = Faker()
        Faker.seed(self.data_config['random_state'])
        np.random.seed(self.data_config['random_state'])
    
    def generate_synthetic_data(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic telecom customer churn data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic customer data
        """
        if n_samples is None:
            n_samples = self.data_config['n_samples']
        
        logger.info(f"Generating {n_samples} synthetic customer records...")
        
        data = {
            'customerID': [f'CUST{i:05d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
            'tenure': np.random.randint(0, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(
                ['No phone service', 'No', 'Yes'], 
                n_samples, 
                p=[0.1, 0.45, 0.45]
            ),
            'InternetService': np.random.choice(
                ['DSL', 'Fiber optic', 'No'], 
                n_samples, 
                p=[0.35, 0.45, 0.2]
            ),
            'OnlineSecurity': np.random.choice(
                ['Yes', 'No', 'No internet service'], 
                n_samples, 
                p=[0.3, 0.5, 0.2]
            ),
            'OnlineBackup': np.random.choice(
                ['Yes', 'No', 'No internet service'], 
                n_samples, 
                p=[0.35, 0.45, 0.2]
            ),
            'DeviceProtection': np.random.choice(
                ['Yes', 'No', 'No internet service'], 
                n_samples, 
                p=[0.35, 0.45, 0.2]
            ),
            'TechSupport': np.random.choice(
                ['Yes', 'No', 'No internet service'], 
                n_samples, 
                p=[0.3, 0.5, 0.2]
            ),
            'StreamingTV': np.random.choice(
                ['Yes', 'No', 'No internet service'], 
                n_samples, 
                p=[0.4, 0.4, 0.2]
            ),
            'StreamingMovies': np.random.choice(
                ['Yes', 'No', 'No internet service'], 
                n_samples, 
                p=[0.4, 0.4, 0.2]
            ),
            'Contract': np.random.choice(
                ['Month-to-month', 'One year', 'Two year'], 
                n_samples, 
                p=[0.55, 0.25, 0.2]
            ),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'PaymentMethod': np.random.choice(
                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                n_samples,
                p=[0.35, 0.20, 0.25, 0.20]
            ),
        }
        
        # Generate monthly charges based on services
        base_charge = 20 + np.random.randn(n_samples) * 5
        service_multiplier = 1.0 + (
            (data['InternetService'] != 'No').astype(int) * 0.5 +
            np.random.choice([0, 0.3, 0.5], n_samples) +
            (data['SeniorCitizen'] == 0).astype(int) * 0.1
        )
        data['MonthlyCharges'] = np.clip(base_charge * service_multiplier, 18.25, 118.75)
        
        # Generate total charges based on tenure and monthly charges
        data['TotalCharges'] = data['MonthlyCharges'] * data['tenure']
        # Some customers might have missing total charges (new customers)
        mask = np.random.random(n_samples) < 0.001
        data['TotalCharges'] = np.where(mask, np.nan, data['TotalCharges'])
        
        # Generate churn label based on multiple factors
        churn_probability = (
            0.05 +  # Base churn rate
            (data['Contract'] == 'Month-to-month').astype(int) * 0.35 +
            (data['tenure'] < 12).astype(int) * 0.25 +
            (data['SeniorCitizen'] == 1).astype(int) * 0.1 +
            (data['MonthlyCharges'] > 80).astype(int) * 0.15 -
            (data['Contract'] == 'Two year').astype(int) * 0.3 -
            (data['tenure'] > 48).astype(int) * 0.2
        )
        churn_probability = np.clip(churn_probability, 0, 0.8)
        data['Churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
        data['Churn'] = data['Churn'].map({0: 'No', 1: 'Yes'})
        
        df = pd.DataFrame(data)
        logger.info(f"Generated dataset with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
        
        return df
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file or generate if not exists.
        
        Args:
            file_path: Path to data file
            
        Returns:
            DataFrame with customer data
        """
        if file_path is None:
            file_path = self.data_config['raw_data_path']
        
        file_path = Path(file_path)
        
        if file_path.exists():
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
        else:
            logger.info(f"Data file not found at {file_path}, generating synthetic data...")
            df = self.generate_synthetic_data()
            
            # Save generated data
            ensure_dir(file_path.parent)
            df.to_csv(file_path, index=False)
            logger.info(f"Saved generated data to {file_path}")
        
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if test_size is None:
            test_size = self.data_config['test_size']
        if val_size is None:
            val_size = self.data_config['val_size']
        if random_state is None:
            random_state = self.data_config['random_state']
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df[self.features_config['target']]
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_df[self.features_config['target']]
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_split_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """
        Save split datasets to disk.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
        """
        save_pickle(train_df, self.data_config['train_path'])
        save_pickle(val_df, self.data_config['val_path'])
        save_pickle(test_df, self.data_config['test_path'])
        
        logger.info(f"Saved split data to {Path(self.data_config['train_path']).parent}")


def main():
    """Main function for data loading."""
    loader = DataLoader()
    
    # Load or generate data
    df = loader.load_data()
    
    # Split and save data
    train_df, val_df, test_df = loader.split_data(df)
    loader.save_split_data(train_df, val_df, test_df)
    
    logger.info("Data loading and splitting completed successfully!")


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    main()
