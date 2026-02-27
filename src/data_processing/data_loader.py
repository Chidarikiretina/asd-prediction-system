"""
Data Loading Module

This module handles loading and initial validation of datasets for ASD prediction.
Supports CSV, Excel, and JSON formats with comprehensive validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union
import logging
import json
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Class for loading and initial validation of ASD screening data.

    Supports multiple file formats and provides data validation,
    stratified splitting, and data quality reporting.
    """

    # Required columns for ASD screening data
    REQUIRED_COLUMNS = ['asd_diagnosis']

    # Expected behavioral features
    BEHAVIORAL_FEATURES = [
        'eye_contact', 'response_to_name', 'pointing',
        'social_smile', 'repetitive_behaviors', 'joint_attention'
    ]

    # Expected demographic features
    DEMOGRAPHIC_FEATURES = ['age_months', 'gender', 'geographic_location']

    def __init__(self, data_path: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to the data directory
            config: Optional configuration dictionary
        """
        self.data_path = Path(data_path)
        self.config = config or {}
        self._validation_errors: List[str] = []
        self._validation_warnings: List[str] = []

    def load_raw_data(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load raw data from file.

        Args:
            filename: Name of the data file
            **kwargs: Additional arguments passed to pandas read functions

        Returns:
            DataFrame containing the raw data

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is not supported
        """
        file_path = self.data_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, **kwargs)
        elif suffix == '.json':
            df = pd.read_json(file_path, **kwargs)
        elif suffix == '.parquet':
            df = pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

    def load_from_multiple_sources(self, filenames: List[str], **kwargs) -> pd.DataFrame:
        """
        Load and concatenate data from multiple files.

        Args:
            filenames: List of filenames to load
            **kwargs: Additional arguments passed to load_raw_data

        Returns:
            Concatenated DataFrame
        """
        dfs = []
        for filename in filenames:
            try:
                df = self.load_raw_data(filename, **kwargs)
                df['_source_file'] = filename
                dfs.append(df)
                logger.info(f"Successfully loaded {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                raise

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(filenames)} files into {len(combined_df)} total records")
        return combined_df

    def validate_data(
        self,
        df: pd.DataFrame,
        strict: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate loaded data for ASD prediction requirements.

        Args:
            df: DataFrame to validate
            strict: If True, warnings become errors

        Returns:
            Tuple of (is_valid, validation_report)
        """
        self._validation_errors = []
        self._validation_warnings = []

        report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'columns_found': list(df.columns),
            'errors': [],
            'warnings': [],
            'missing_values': {},
            'data_types': {}
        }

        # Check for empty DataFrame
        if len(df) == 0:
            self._validation_errors.append("DataFrame is empty")

        # Check for required columns
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                self._validation_errors.append(f"Required column missing: {col}")

        # Check for behavioral features
        missing_behavioral = [f for f in self.BEHAVIORAL_FEATURES if f not in df.columns]
        if missing_behavioral:
            self._validation_warnings.append(
                f"Missing behavioral features: {missing_behavioral}"
            )

        # Check for demographic features
        missing_demographic = [f for f in self.DEMOGRAPHIC_FEATURES if f not in df.columns]
        if missing_demographic:
            self._validation_warnings.append(
                f"Missing demographic features: {missing_demographic}"
            )

        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        for col in df.columns:
            if missing_counts[col] > 0:
                report['missing_values'][col] = {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_pct[col])
                }
                if missing_pct[col] > 50:
                    self._validation_warnings.append(
                        f"Column '{col}' has {missing_pct[col]}% missing values"
                    )

        # Check data types
        for col in df.columns:
            report['data_types'][col] = str(df[col].dtype)

        # Validate target variable
        if 'asd_diagnosis' in df.columns:
            unique_values = df['asd_diagnosis'].dropna().unique()
            if len(unique_values) > 2:
                self._validation_errors.append(
                    f"Target variable 'asd_diagnosis' should be binary, found {len(unique_values)} unique values"
                )

            # Check class balance
            if len(df) > 0:
                value_counts = df['asd_diagnosis'].value_counts(normalize=True)
                min_class_pct = value_counts.min() * 100
                if min_class_pct < 10:
                    self._validation_warnings.append(
                        f"Imbalanced classes detected: minority class is {min_class_pct:.1f}%"
                    )

        # Validate age range (if present)
        if 'age_months' in df.columns:
            age_min = df['age_months'].min()
            age_max = df['age_months'].max()
            if age_min < 0:
                self._validation_errors.append(f"Invalid age values: minimum is {age_min}")
            if age_max > 120:
                self._validation_warnings.append(
                    f"Age values exceed expected range: maximum is {age_max} months"
                )

        # Check for duplicate records
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self._validation_warnings.append(f"Found {duplicates} duplicate records")

        # Compile report
        report['errors'] = self._validation_errors.copy()
        report['warnings'] = self._validation_warnings.copy()

        if strict:
            is_valid = len(self._validation_errors) == 0 and len(self._validation_warnings) == 0
        else:
            is_valid = len(self._validation_errors) == 0

        # Log results
        if self._validation_errors:
            for error in self._validation_errors:
                logger.error(f"Validation error: {error}")
        if self._validation_warnings:
            for warning in self._validation_warnings:
                logger.warning(f"Validation warning: {warning}")

        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.error("Data validation failed")

        return is_valid, report

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'asd_diagnosis',
        test_size: float = 0.2,
        validation_size: Optional[float] = None,
        random_state: int = 42,
        stratify: bool = True
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Split data into train, test, and optionally validation sets.

        Args:
            df: Input DataFrame
            target_column: Name of the target column for stratification
            test_size: Proportion of data for test set
            validation_size: Optional proportion for validation set (from training data)
            random_state: Random seed for reproducibility
            stratify: Whether to use stratified splitting

        Returns:
            Tuple of (train_df, test_df) or (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        logger.info(f"Splitting data with test_size={test_size}, validation_size={validation_size}")

        # Determine stratification
        stratify_col = df[target_column] if stratify and target_column in df.columns else None

        # Initial split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        logger.info(f"Train+Val: {len(train_val_df)}, Test: {len(test_df)}")

        if validation_size is not None and validation_size > 0:
            # Split train into train and validation
            # Adjust validation size relative to train+val
            val_ratio = validation_size / (1 - test_size)

            stratify_col = train_val_df[target_column] if stratify and target_column in train_val_df.columns else None

            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_ratio,
                random_state=random_state,
                stratify=stratify_col
            )

            logger.info(f"Final split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return train_df, val_df, test_df

        logger.info(f"Final split - Train: {len(train_val_df)}, Test: {len(test_df)}")
        return train_val_df, test_df

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing data summary statistics
        """
        summary = {
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_summary'][col] = {
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'std': float(df[col].std()) if not df[col].isna().all() else None,
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None,
                'median': float(df[col].median()) if not df[col].isna().all() else None
            }

        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            summary['categorical_summary'][col] = {
                'unique_values': int(df[col].nunique()),
                'top_values': value_counts.head(5).to_dict(),
                'mode': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
            }

        # Target variable distribution (if present)
        if 'asd_diagnosis' in df.columns:
            summary['target_distribution'] = df['asd_diagnosis'].value_counts().to_dict()
            summary['target_balance'] = df['asd_diagnosis'].value_counts(normalize=True).to_dict()

        return summary

    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save processed data to file.

        Args:
            df: DataFrame to save
            filename: Output filename
            output_path: Optional output directory (defaults to processed data path)

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = self.data_path.parent / 'processed'
        else:
            output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / filename

        suffix = file_path.suffix.lower()

        if suffix == '.csv':
            df.to_csv(file_path, index=False)
        elif suffix in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=False)
        elif suffix == '.json':
            df.to_json(file_path, orient='records', indent=2)
        elif suffix == '.parquet':
            df.to_parquet(file_path, index=False)
        else:
            # Default to CSV
            file_path = file_path.with_suffix('.csv')
            df.to_csv(file_path, index=False)

        logger.info(f"Saved processed data to {file_path}")
        return file_path


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    loader = DataLoader("data/raw")

    # Example: Load and validate data
    # data = loader.load_raw_data("asd_data.csv")
    # is_valid, report = loader.validate_data(data)
    # print(f"Validation passed: {is_valid}")
    # print(f"Report: {report}")
