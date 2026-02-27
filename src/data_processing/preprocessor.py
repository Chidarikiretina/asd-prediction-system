"""
Data Preprocessing Module

This module handles data cleaning, missing value imputation, encoding,
normalization, and outlier detection for ASD screening data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Class for preprocessing ASD screening data.

    Handles missing values, categorical encoding, feature normalization,
    outlier detection, and feature interactions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with default settings.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoder: Optional[OneHotEncoder] = None
        self.imputers: Dict[str, SimpleImputer] = {}
        self.fitted = False
        self._numeric_columns: List[str] = []
        self._categorical_columns: List[str] = []
        self._binary_columns: List[str] = []
        self._feature_stats: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, target_column: str = 'asd_diagnosis') -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.

        Args:
            df: Training DataFrame
            target_column: Name of target column to exclude from preprocessing

        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessor on training data")

        # Identify column types
        self._identify_column_types(df, target_column)

        # Store feature statistics for outlier detection
        self._compute_feature_stats(df)

        self.fitted = True
        logger.info(f"Preprocessor fitted. Numeric: {len(self._numeric_columns)}, "
                    f"Categorical: {len(self._categorical_columns)}, "
                    f"Binary: {len(self._binary_columns)}")

        return self

    def _identify_column_types(self, df: pd.DataFrame, target_column: str) -> None:
        """Identify numeric, categorical, and binary columns."""
        feature_cols = [col for col in df.columns if col != target_column]

        for col in feature_cols:
            if df[col].dtype in ['object', 'category']:
                unique_vals = df[col].nunique()
                if unique_vals == 2:
                    self._binary_columns.append(col)
                else:
                    self._categorical_columns.append(col)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                unique_vals = df[col].nunique()
                if unique_vals == 2 and set(df[col].dropna().unique()).issubset({0, 1}):
                    self._binary_columns.append(col)
                else:
                    self._numeric_columns.append(col)

    def _compute_feature_stats(self, df: pd.DataFrame) -> None:
        """Compute and store statistics for numeric features."""
        for col in self._numeric_columns:
            if col in df.columns:
                self._feature_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'median': df[col].median(),
                    'q1': df[col].quantile(0.25),
                    'q3': df[col].quantile(0.75),
                    'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
                }

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'median',
        categorical_strategy: str = 'most_frequent',
        knn_neighbors: int = 5
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy for numeric columns ('mean', 'median', 'most_frequent', 'knn')
            categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')
            knn_neighbors: Number of neighbors for KNN imputation

        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        logger.info(f"Handling missing values with strategy: {strategy}")

        missing_before = df.isnull().sum().sum()

        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            if strategy == 'knn':
                if 'numeric' not in self.imputers:
                    self.imputers['numeric'] = KNNImputer(n_neighbors=knn_neighbors)
                    df[numeric_cols] = self.imputers['numeric'].fit_transform(df[numeric_cols])
                else:
                    df[numeric_cols] = self.imputers['numeric'].transform(df[numeric_cols])
            else:
                if 'numeric' not in self.imputers:
                    self.imputers['numeric'] = SimpleImputer(strategy=strategy)
                    df[numeric_cols] = self.imputers['numeric'].fit_transform(df[numeric_cols])
                else:
                    df[numeric_cols] = self.imputers['numeric'].transform(df[numeric_cols])

        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            if 'categorical' not in self.imputers:
                fill_value = 'Unknown' if categorical_strategy == 'constant' else None
                self.imputers['categorical'] = SimpleImputer(
                    strategy='most_frequent' if categorical_strategy != 'constant' else 'constant',
                    fill_value=fill_value
                )
                df[categorical_cols] = self.imputers['categorical'].fit_transform(df[categorical_cols])
            else:
                df[categorical_cols] = self.imputers['categorical'].transform(df[categorical_cols])

        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values reduced from {missing_before} to {missing_after}")

        return df

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        method: str = 'auto',
        drop_first: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            df: Input DataFrame
            method: Encoding method ('auto', 'label', 'onehot')
                   'auto' uses label encoding for binary, one-hot for multi-class
            drop_first: Whether to drop first category in one-hot encoding

        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        logger.info(f"Encoding categorical features with method: {method}")

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_cols:
            unique_vals = df[col].nunique()

            if method == 'label' or (method == 'auto' and unique_vals == 2):
                # Label encoding for binary or when explicitly requested
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Handle unseen values by fitting on all unique values
                    df[col] = df[col].fillna('Unknown')
                    self.label_encoders[col].fit(df[col])

                df[col] = self.label_encoders[col].transform(df[col])
                logger.debug(f"Label encoded column: {col}")

            elif method == 'onehot' or (method == 'auto' and unique_vals > 2):
                # One-hot encoding for multi-class
                df[col] = df[col].fillna('Unknown')
                dummies = pd.get_dummies(
                    df[col],
                    prefix=col,
                    drop_first=drop_first,
                    dtype=int
                )
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                logger.debug(f"One-hot encoded column: {col} -> {list(dummies.columns)}")

        logger.info(f"Encoding complete. New shape: {df.shape}")
        return df

    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Normalize numerical features.

        Args:
            df: Input DataFrame
            columns: List of columns to normalize (None for all numeric)
            method: Normalization method ('standard', 'minmax')

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        logger.info(f"Normalizing features with method: {method}")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude binary columns (0/1 values)
            columns = [col for col in columns if not (
                df[col].nunique() == 2 and
                set(df[col].dropna().unique()).issubset({0, 1})
            )]

        if not columns:
            logger.warning("No columns to normalize")
            return df

        if self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            df[columns] = self.scaler.fit_transform(df[columns])
        else:
            df[columns] = self.scaler.transform(df[columns])

        logger.info(f"Normalized {len(columns)} columns")
        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 1.5,
        columns: Optional[List[str]] = None,
        handle_method: str = 'clip'
    ) -> pd.DataFrame:
        """
        Remove or handle outliers in numeric features.

        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection (1.5 for IQR, 3 for z-score)
            columns: Columns to check (None for all numeric)
            handle_method: How to handle outliers ('remove', 'clip', 'nan')

        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        logger.info(f"Handling outliers using {method} method with {handle_method}")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_count = 0

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'iqr':
                # Use stored stats if available, otherwise compute
                if col in self._feature_stats:
                    q1 = self._feature_stats[col]['q1']
                    q3 = self._feature_stats[col]['q3']
                    iqr = self._feature_stats[col]['iqr']
                else:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

            elif method == 'zscore':
                if col in self._feature_stats:
                    mean = self._feature_stats[col]['mean']
                    std = self._feature_stats[col]['std']
                else:
                    mean = df[col].mean()
                    std = df[col].std()

                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

            else:
                raise ValueError(f"Unknown outlier method: {method}")

            # Identify outliers
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            col_outliers = outliers.sum()
            outlier_count += col_outliers

            if col_outliers > 0:
                if handle_method == 'remove':
                    df = df[~outliers]
                elif handle_method == 'clip':
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                elif handle_method == 'nan':
                    df.loc[outliers, col] = np.nan

                logger.debug(f"Column {col}: {col_outliers} outliers handled")

        logger.info(f"Total outliers handled: {outlier_count}")
        return df

    def create_feature_interactions(
        self,
        df: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]],
        operations: List[str] = None
    ) -> pd.DataFrame:
        """
        Create interaction features between specified columns.

        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing feature pairs
            operations: List of operations ('multiply', 'add', 'subtract', 'ratio')
                       Default is ['multiply']

        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()
        logger.info(f"Creating feature interactions for {len(feature_pairs)} pairs")

        if operations is None:
            operations = ['multiply']

        for feat1, feat2 in feature_pairs:
            if feat1 not in df.columns or feat2 not in df.columns:
                logger.warning(f"Skipping interaction: {feat1} or {feat2} not in DataFrame")
                continue

            for op in operations:
                if op == 'multiply':
                    col_name = f"{feat1}_x_{feat2}"
                    df[col_name] = df[feat1] * df[feat2]
                elif op == 'add':
                    col_name = f"{feat1}_plus_{feat2}"
                    df[col_name] = df[feat1] + df[feat2]
                elif op == 'subtract':
                    col_name = f"{feat1}_minus_{feat2}"
                    df[col_name] = df[feat1] - df[feat2]
                elif op == 'ratio':
                    col_name = f"{feat1}_div_{feat2}"
                    df[col_name] = df[feat1] / (df[feat2] + 1e-8)  # Avoid division by zero

                logger.debug(f"Created interaction feature: {col_name}")

        logger.info(f"Created {len(feature_pairs) * len(operations)} interaction features")
        return df

    def create_aggregate_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        aggregations: List[str] = None
    ) -> pd.DataFrame:
        """
        Create aggregate features from multiple columns.

        Args:
            df: Input DataFrame
            columns: List of columns to aggregate
            aggregations: List of aggregation functions ('sum', 'mean', 'std', 'min', 'max')

        Returns:
            DataFrame with aggregate features added
        """
        df = df.copy()

        if aggregations is None:
            aggregations = ['sum', 'mean']

        existing_cols = [col for col in columns if col in df.columns]
        if not existing_cols:
            logger.warning("No valid columns for aggregation")
            return df

        prefix = '_'.join(existing_cols[:3])
        if len(existing_cols) > 3:
            prefix += '_etc'

        for agg in aggregations:
            if agg == 'sum':
                df[f'{prefix}_sum'] = df[existing_cols].sum(axis=1)
            elif agg == 'mean':
                df[f'{prefix}_mean'] = df[existing_cols].mean(axis=1)
            elif agg == 'std':
                df[f'{prefix}_std'] = df[existing_cols].std(axis=1)
            elif agg == 'min':
                df[f'{prefix}_min'] = df[existing_cols].min(axis=1)
            elif agg == 'max':
                df[f'{prefix}_max'] = df[existing_cols].max(axis=1)

        logger.info(f"Created {len(aggregations)} aggregate features from {len(existing_cols)} columns")
        return df

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str = 'asd_diagnosis',
        missing_strategy: str = 'median',
        normalize: bool = True,
        handle_outliers: bool = True,
        create_interactions: bool = False,
        interaction_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run the complete preprocessing pipeline.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            missing_strategy: Strategy for missing values
            normalize: Whether to normalize features
            handle_outliers: Whether to handle outliers
            create_interactions: Whether to create interaction features
            interaction_pairs: Feature pairs for interactions

        Returns:
            Tuple of (preprocessed features DataFrame, target Series)
        """
        logger.info("Starting preprocessing pipeline")

        # Separate features and target
        if target_column in df.columns:
            y = df[target_column].copy()
            X = df.drop(columns=[target_column])
        else:
            y = None
            X = df.copy()

        # Fit if not already fitted
        if not self.fitted:
            self.fit(df, target_column)

        # Step 1: Handle missing values
        X = self.handle_missing_values(X, strategy=missing_strategy)

        # Step 2: Handle outliers (before encoding)
        if handle_outliers:
            X = self.remove_outliers(X, handle_method='clip')

        # Step 3: Create interaction features (before encoding)
        if create_interactions and interaction_pairs:
            X = self.create_feature_interactions(X, interaction_pairs)

        # Step 4: Encode categorical features
        X = self.encode_categorical_features(X, method='auto')

        # Step 5: Normalize features
        if normalize:
            X = self.normalize_features(X, method='standard')

        logger.info(f"Preprocessing complete. Final shape: {X.shape}")
        return X, y

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save preprocessor state to file.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'config': self.config,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputers': self.imputers,
            'fitted': self.fitted,
            'numeric_columns': self._numeric_columns,
            'categorical_columns': self._categorical_columns,
            'binary_columns': self._binary_columns,
            'feature_stats': self._feature_stats
        }

        joblib.dump(state, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'DataPreprocessor':
        """
        Load preprocessor state from file.

        Args:
            filepath: Path to saved file

        Returns:
            Loaded DataPreprocessor instance
        """
        state = joblib.load(filepath)

        preprocessor = cls(config=state['config'])
        preprocessor.scaler = state['scaler']
        preprocessor.label_encoders = state['label_encoders']
        preprocessor.imputers = state['imputers']
        preprocessor.fitted = state['fitted']
        preprocessor._numeric_columns = state['numeric_columns']
        preprocessor._categorical_columns = state['categorical_columns']
        preprocessor._binary_columns = state['binary_columns']
        preprocessor._feature_stats = state['feature_stats']

        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    preprocessor = DataPreprocessor()

    # Example: Create sample data and preprocess
    # sample_data = pd.DataFrame({
    #     'age_months': [24, 30, 18, 36, 28],
    #     'gender': ['M', 'F', 'M', 'F', 'M'],
    #     'eye_contact': [1, 0, 1, 1, 0],
    #     'asd_diagnosis': [0, 1, 0, 0, 1]
    # })
    # X, y = preprocessor.preprocess_pipeline(sample_data)
    # print(f"Preprocessed shape: {X.shape}")
