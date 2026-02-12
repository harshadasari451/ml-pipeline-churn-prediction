"""
Data validation module for ensuring data quality.
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Data validator class for checking data quality and consistency.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataValidator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.features_config = self.config['features']
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has expected columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        expected_columns = (
            self.features_config['numerical'] +
            self.features_config['categorical'] +
            [self.features_config['target']]
        )
        
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
        
        extra_columns = set(df.columns) - set(expected_columns) - {'customerID'}
        if extra_columns:
            logger.warning(f"Extra columns found (will be ignored): {extra_columns}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data types of columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Check numerical columns
        for col in self.features_config['numerical']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' should be numeric but is {df[col].dtype}")
        
        # Check categorical columns
        for col in self.features_config['categorical']:
            if col in df.columns:
                # Allow both string and numeric types for categorical
                if not (pd.api.types.is_object_dtype(df[col]) or 
                       pd.api.types.is_categorical_dtype(df[col]) or
                       pd.api.types.is_integer_dtype(df[col])):
                    errors.append(f"Column '{col}' should be categorical but is {df[col].dtype}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_missing_values(
        self, 
        df: pd.DataFrame, 
        max_missing_percent: float = 0.2
    ) -> Tuple[bool, List[str]]:
        """
        Check for missing values.
        
        Args:
            df: DataFrame to validate
            max_missing_percent: Maximum allowed percentage of missing values
            
        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []
        
        missing_stats = df.isnull().sum()
        missing_percent = (missing_stats / len(df)) * 100
        
        for col, pct in missing_percent.items():
            if pct > 0:
                if pct > max_missing_percent * 100:
                    warnings.append(
                        f"Column '{col}' has {pct:.2f}% missing values (threshold: {max_missing_percent*100}%)"
                    )
                else:
                    logger.info(f"Column '{col}' has {pct:.2f}% missing values")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def validate_target_distribution(
        self, 
        df: pd.DataFrame,
        min_class_percent: float = 0.05
    ) -> Tuple[bool, List[str]]:
        """
        Validate target variable distribution.
        
        Args:
            df: DataFrame to validate
            min_class_percent: Minimum percentage for each class
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        target_col = self.features_config['target']
        
        if target_col not in df.columns:
            errors.append(f"Target column '{target_col}' not found")
            return False, errors
        
        # Check class distribution
        class_counts = df[target_col].value_counts()
        class_percent = (class_counts / len(df)) * 100
        
        logger.info(f"Target distribution:\n{class_counts}")
        
        for cls, pct in class_percent.items():
            if pct < min_class_percent * 100:
                errors.append(
                    f"Class '{cls}' has only {pct:.2f}% of samples (minimum: {min_class_percent*100}%)"
                )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_numerical_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate numerical column ranges.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []
        
        for col in self.features_config['numerical']:
            if col not in df.columns:
                continue
            
            # Check for negative values where they shouldn't exist
            if col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                if (df[col] < 0).any():
                    warnings.append(f"Column '{col}' has negative values")
            
            # Check for outliers (values beyond 3 standard deviations)
            mean = df[col].mean()
            std = df[col].std()
            outliers = ((df[col] - mean).abs() > 3 * std).sum()
            if outliers > 0:
                outlier_pct = (outliers / len(df)) * 100
                if outlier_pct > 5:
                    warnings.append(
                        f"Column '{col}' has {outliers} outliers ({outlier_pct:.2f}% of data)"
                    )
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Running data validation...")
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Schema validation
        schema_valid, schema_errors = self.validate_schema(df)
        if not schema_valid:
            results['is_valid'] = False
            results['errors'].extend(schema_errors)
        
        # Data type validation
        dtype_valid, dtype_errors = self.validate_data_types(df)
        if not dtype_valid:
            results['is_valid'] = False
            results['errors'].extend(dtype_errors)
        
        # Missing values validation
        missing_valid, missing_warnings = self.validate_missing_values(df)
        results['warnings'].extend(missing_warnings)
        
        # Target distribution validation
        target_valid, target_errors = self.validate_target_distribution(df)
        if not target_valid:
            results['is_valid'] = False
            results['errors'].extend(target_errors)
        
        # Numerical ranges validation
        ranges_valid, ranges_warnings = self.validate_numerical_ranges(df)
        results['warnings'].extend(ranges_warnings)
        
        # Log results
        if results['is_valid']:
            logger.info("✓ Data validation passed")
        else:
            logger.error(f"✗ Data validation failed with {len(results['errors'])} errors")
            for error in results['errors']:
                logger.error(f"  - {error}")
        
        if results['warnings']:
            logger.warning(f"Data validation completed with {len(results['warnings'])} warnings")
            for warning in results['warnings']:
                logger.warning(f"  - {warning}")
        
        return results


def main():
    """Main function for data validation."""
    from src.data.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    
    # Validate data
    validator = DataValidator()
    results = validator.validate_data(df)
    
    if results['is_valid']:
        logger.info("Data validation completed successfully!")
    else:
        logger.error("Data validation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    exit(main())
