"""
Unit Tests for Data Preprocessor Module

Tests for data preprocessing, encoding, normalization, and outlier handling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data_processing.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100

        data = {
            'age_months': np.random.randint(18, 37, n_samples).astype(float),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'geographic_location': np.random.choice(['Harare', 'Bulawayo', 'Manicaland'], n_samples),
            'eye_contact': np.random.randint(0, 2, n_samples),
            'response_to_name': np.random.randint(0, 2, n_samples),
            'mchat_score': np.random.randint(0, 20, n_samples).astype(float),
            'word_count': np.random.randint(0, 300, n_samples).astype(float),
            'asd_diagnosis': np.random.randint(0, 2, n_samples)
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def data_with_missing(self, sample_data):
        """Create data with missing values."""
        df = sample_data.copy()
        # Introduce missing values
        df.loc[0:5, 'age_months'] = np.nan
        df.loc[10:15, 'mchat_score'] = np.nan
        df.loc[20:22, 'gender'] = np.nan
        return df

    @pytest.fixture
    def data_with_outliers(self, sample_data):
        """Create data with outliers."""
        df = sample_data.copy()
        # Add outliers to age_months
        df.loc[0, 'age_months'] = 100  # Outlier (should be 18-36)
        df.loc[1, 'age_months'] = -5   # Outlier
        df.loc[2, 'word_count'] = 5000  # Outlier
        return df

    def test_init(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is None
        assert preprocessor.label_encoders == {}
        assert preprocessor.fitted is False

    def test_fit(self, sample_data):
        """Test fitting preprocessor on data."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_data, target_column='asd_diagnosis')

        assert preprocessor.fitted is True
        assert len(preprocessor._numeric_columns) > 0
        assert 'age_months' in preprocessor._numeric_columns

    def test_handle_missing_values_median(self, data_with_missing):
        """Test handling missing values with median strategy."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(
            data_with_missing,
            strategy='median'
        )

        # Check no missing values in numeric columns
        assert df_clean['age_months'].isna().sum() == 0
        assert df_clean['mchat_score'].isna().sum() == 0

    def test_handle_missing_values_mean(self, data_with_missing):
        """Test handling missing values with mean strategy."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(
            data_with_missing,
            strategy='mean'
        )

        assert df_clean['age_months'].isna().sum() == 0

    def test_handle_missing_values_categorical(self, data_with_missing):
        """Test handling missing values in categorical columns."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(data_with_missing)

        # Check categorical column is imputed
        assert df_clean['gender'].isna().sum() == 0

    def test_encode_categorical_features_auto(self, sample_data):
        """Test automatic categorical encoding."""
        preprocessor = DataPreprocessor()
        df_encoded = preprocessor.encode_categorical_features(sample_data, method='auto')

        # Gender should be label encoded (binary)
        assert df_encoded['gender'].dtype in [np.int64, np.int32, int]

        # Geographic location should be one-hot encoded (multi-class)
        assert 'geographic_location' not in df_encoded.columns
        # Check for one-hot columns
        onehot_cols = [c for c in df_encoded.columns if c.startswith('geographic_location_')]
        assert len(onehot_cols) > 0

    def test_encode_categorical_features_label(self, sample_data):
        """Test label encoding."""
        preprocessor = DataPreprocessor()
        df_encoded = preprocessor.encode_categorical_features(sample_data, method='label')

        assert df_encoded['gender'].dtype in [np.int64, np.int32, int]
        assert set(df_encoded['gender'].unique()).issubset({0, 1})

    def test_normalize_features_standard(self, sample_data):
        """Test standard normalization."""
        preprocessor = DataPreprocessor()

        # Only normalize numeric columns
        columns_to_normalize = ['age_months', 'mchat_score', 'word_count']
        df_norm = preprocessor.normalize_features(
            sample_data,
            columns=columns_to_normalize,
            method='standard'
        )

        # Check normalized columns have ~mean 0 and ~std 1
        for col in columns_to_normalize:
            assert abs(df_norm[col].mean()) < 0.1
            assert abs(df_norm[col].std() - 1) < 0.1

    def test_normalize_features_minmax(self, sample_data):
        """Test min-max normalization."""
        preprocessor = DataPreprocessor()

        columns_to_normalize = ['age_months', 'mchat_score']
        df_norm = preprocessor.normalize_features(
            sample_data,
            columns=columns_to_normalize,
            method='minmax'
        )

        # Check normalized columns are in [0, 1] range
        for col in columns_to_normalize:
            assert df_norm[col].min() >= 0
            assert df_norm[col].max() <= 1

    def test_remove_outliers_iqr_clip(self, data_with_outliers):
        """Test outlier removal using IQR method with clipping."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(data_with_outliers, 'asd_diagnosis')

        df_clean = preprocessor.remove_outliers(
            data_with_outliers,
            method='iqr',
            handle_method='clip',
            columns=['age_months', 'word_count']
        )

        # Outliers should be clipped, not removed
        assert len(df_clean) == len(data_with_outliers)

        # Values should be within reasonable bounds
        assert df_clean['age_months'].min() >= 0

    def test_remove_outliers_iqr_remove(self, data_with_outliers):
        """Test outlier removal using IQR method with removal."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(data_with_outliers, 'asd_diagnosis')

        df_clean = preprocessor.remove_outliers(
            data_with_outliers,
            method='iqr',
            handle_method='remove',
            columns=['age_months']
        )

        # Some rows should be removed
        assert len(df_clean) < len(data_with_outliers)

    def test_remove_outliers_zscore(self, data_with_outliers):
        """Test outlier removal using z-score method."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(data_with_outliers, 'asd_diagnosis')

        df_clean = preprocessor.remove_outliers(
            data_with_outliers,
            method='zscore',
            threshold=3,
            handle_method='clip'
        )

        assert len(df_clean) == len(data_with_outliers)

    def test_create_feature_interactions(self, sample_data):
        """Test feature interaction creation."""
        preprocessor = DataPreprocessor()

        feature_pairs = [
            ('eye_contact', 'response_to_name'),
            ('age_months', 'mchat_score')
        ]

        df_interactions = preprocessor.create_feature_interactions(
            sample_data,
            feature_pairs,
            operations=['multiply', 'add']
        )

        # Check interaction columns were created
        assert 'eye_contact_x_response_to_name' in df_interactions.columns
        assert 'age_months_plus_mchat_score' in df_interactions.columns

    def test_create_feature_interactions_missing_column(self, sample_data):
        """Test feature interaction with missing column."""
        preprocessor = DataPreprocessor()

        feature_pairs = [
            ('eye_contact', 'nonexistent_column')
        ]

        df_interactions = preprocessor.create_feature_interactions(
            sample_data,
            feature_pairs
        )

        # Should handle gracefully without creating interaction
        assert 'eye_contact_x_nonexistent_column' not in df_interactions.columns

    def test_create_aggregate_features(self, sample_data):
        """Test aggregate feature creation."""
        preprocessor = DataPreprocessor()

        columns = ['eye_contact', 'response_to_name']
        df_agg = preprocessor.create_aggregate_features(
            sample_data,
            columns,
            aggregations=['sum', 'mean']
        )

        # Check aggregate columns were created
        assert any('sum' in col for col in df_agg.columns)
        assert any('mean' in col for col in df_agg.columns)

    def test_preprocess_pipeline(self, data_with_missing):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor()

        X, y = preprocessor.preprocess_pipeline(
            data_with_missing,
            target_column='asd_diagnosis',
            missing_strategy='median',
            normalize=True,
            handle_outliers=True
        )

        # Check output
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert 'asd_diagnosis' not in X.columns
        assert len(X) == len(y)

        # Check no missing values
        assert X.isnull().sum().sum() == 0

    def test_save_and_load(self, sample_data):
        """Test saving and loading preprocessor state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'preprocessor.joblib'

            # Fit and save
            preprocessor = DataPreprocessor()
            preprocessor.fit(sample_data, 'asd_diagnosis')
            preprocessor.handle_missing_values(sample_data)
            preprocessor.save(filepath)

            # Load and verify
            loaded = DataPreprocessor.load(filepath)

            assert loaded.fitted == preprocessor.fitted
            assert loaded._numeric_columns == preprocessor._numeric_columns

    def test_preprocessor_consistency(self, sample_data):
        """Test that preprocessor transforms consistently after fitting."""
        preprocessor = DataPreprocessor()

        # First transformation
        X1, y1 = preprocessor.preprocess_pipeline(sample_data, normalize=True)

        # Second transformation on same data should give same result
        # Reset preprocessor state
        preprocessor2 = DataPreprocessor()
        X2, y2 = preprocessor2.preprocess_pipeline(sample_data, normalize=True)

        # Shapes should match
        assert X1.shape == X2.shape


class TestDataPreprocessorEdgeCases:
    """Test edge cases for DataPreprocessor."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame()

        # Should handle gracefully
        df_processed = preprocessor.handle_missing_values(df)
        assert len(df_processed) == 0

    def test_single_row(self):
        """Test handling of single row DataFrame."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'age_months': [24],
            'gender': ['M'],
            'asd_diagnosis': [0]
        })

        df_processed = preprocessor.handle_missing_values(df)
        assert len(df_processed) == 1

    def test_all_missing_column(self):
        """Test handling of column with all missing values."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'age_months': [np.nan, np.nan, np.nan],
            'gender': ['M', 'F', 'M'],
            'asd_diagnosis': [0, 1, 0]
        })

        # Should handle without error
        df_processed = preprocessor.handle_missing_values(df)
        assert len(df_processed) == 3

    def test_no_categorical_columns(self):
        """Test encoding when no categorical columns exist."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'age_months': [24, 30, 18],
            'score': [5, 10, 8],
            'asd_diagnosis': [0, 1, 0]
        })

        df_encoded = preprocessor.encode_categorical_features(df)

        # Should return same DataFrame
        assert list(df_encoded.columns) == list(df.columns)

    def test_no_numeric_columns(self):
        """Test normalization when no numeric columns exist."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'gender': ['M', 'F', 'M'],
            'location': ['A', 'B', 'C']
        })

        df_norm = preprocessor.normalize_features(df)

        # Should return same DataFrame
        assert list(df_norm.columns) == list(df.columns)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
