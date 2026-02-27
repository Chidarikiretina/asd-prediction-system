"""
Unit Tests for Data Loader Module

Tests for data loading, validation, and splitting functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data_processing.data_loader import DataLoader, load_config


class TestDataLoader:
    """Test cases for DataLoader class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample ASD screening data."""
        np.random.seed(42)
        n_samples = 100

        data = {
            'participant_id': [f'P{i:04d}' for i in range(n_samples)],
            'age_months': np.random.randint(18, 37, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'geographic_location': np.random.choice(['Harare', 'Bulawayo', 'Manicaland'], n_samples),
            'eye_contact': np.random.randint(0, 2, n_samples),
            'response_to_name': np.random.randint(0, 2, n_samples),
            'pointing': np.random.randint(0, 2, n_samples),
            'social_smile': np.random.randint(0, 2, n_samples),
            'repetitive_behaviors': np.random.randint(0, 2, n_samples),
            'joint_attention': np.random.randint(0, 2, n_samples),
            'asd_diagnosis': np.random.randint(0, 2, n_samples)
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def temp_data_dir(self, sample_data):
        """Create temporary directory with test data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save CSV
            csv_path = Path(tmpdir) / 'test_data.csv'
            sample_data.to_csv(csv_path, index=False)

            # Save JSON
            json_path = Path(tmpdir) / 'test_data.json'
            sample_data.to_json(json_path, orient='records')

            yield tmpdir

    def test_init(self, temp_data_dir):
        """Test DataLoader initialization."""
        loader = DataLoader(temp_data_dir)
        assert loader.data_path == Path(temp_data_dir)
        assert loader.config == {}

    def test_load_csv(self, temp_data_dir):
        """Test loading CSV file."""
        loader = DataLoader(temp_data_dir)
        df = loader.load_raw_data('test_data.csv')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert 'asd_diagnosis' in df.columns

    def test_load_json(self, temp_data_dir):
        """Test loading JSON file."""
        loader = DataLoader(temp_data_dir)
        df = loader.load_raw_data('test_data.json')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_load_nonexistent_file(self, temp_data_dir):
        """Test loading non-existent file raises error."""
        loader = DataLoader(temp_data_dir)

        with pytest.raises(FileNotFoundError):
            loader.load_raw_data('nonexistent.csv')

    def test_load_unsupported_format(self, temp_data_dir):
        """Test loading unsupported file format raises error."""
        # Create a dummy file with unsupported extension
        dummy_path = Path(temp_data_dir) / 'test.xyz'
        dummy_path.write_text('dummy content')

        loader = DataLoader(temp_data_dir)

        with pytest.raises(ValueError):
            loader.load_raw_data('test.xyz')

    def test_validate_data_valid(self, sample_data, temp_data_dir):
        """Test validation passes for valid data."""
        loader = DataLoader(temp_data_dir)
        is_valid, report = loader.validate_data(sample_data)

        assert is_valid is True
        assert report['total_records'] == 100
        assert 'asd_diagnosis' in report['columns_found']
        assert len(report['errors']) == 0

    def test_validate_data_missing_target(self, sample_data, temp_data_dir):
        """Test validation fails when target column is missing."""
        df = sample_data.drop(columns=['asd_diagnosis'])
        loader = DataLoader(temp_data_dir)

        is_valid, report = loader.validate_data(df)

        assert is_valid is False
        assert any('asd_diagnosis' in error for error in report['errors'])

    def test_validate_data_empty_dataframe(self, temp_data_dir):
        """Test validation fails for empty DataFrame."""
        df = pd.DataFrame()
        loader = DataLoader(temp_data_dir)

        is_valid, report = loader.validate_data(df)

        assert is_valid is False
        assert any('empty' in error.lower() for error in report['errors'])

    def test_validate_data_with_missing_values(self, sample_data, temp_data_dir):
        """Test validation reports missing values."""
        df = sample_data.copy()
        df.loc[0:10, 'age_months'] = np.nan

        loader = DataLoader(temp_data_dir)
        is_valid, report = loader.validate_data(df)

        assert 'age_months' in report['missing_values']
        assert report['missing_values']['age_months']['count'] == 11

    def test_split_data_basic(self, sample_data, temp_data_dir):
        """Test basic train/test split."""
        loader = DataLoader(temp_data_dir)
        train_df, test_df = loader.split_data(sample_data, test_size=0.2)

        assert len(train_df) == 80
        assert len(test_df) == 20
        assert len(train_df) + len(test_df) == len(sample_data)

    def test_split_data_with_validation(self, sample_data, temp_data_dir):
        """Test train/validation/test split."""
        loader = DataLoader(temp_data_dir)
        train_df, val_df, test_df = loader.split_data(
            sample_data,
            test_size=0.2,
            validation_size=0.1
        )

        assert len(train_df) + len(val_df) + len(test_df) == len(sample_data)
        assert len(test_df) == 20  # 20% of 100
        # Validation is 10% of original, which is 10/80 of train+val

    def test_split_data_stratified(self, sample_data, temp_data_dir):
        """Test stratified splitting maintains class balance."""
        loader = DataLoader(temp_data_dir)
        train_df, test_df = loader.split_data(sample_data, stratify=True)

        original_ratio = sample_data['asd_diagnosis'].mean()
        train_ratio = train_df['asd_diagnosis'].mean()
        test_ratio = test_df['asd_diagnosis'].mean()

        # Ratios should be approximately equal
        assert abs(train_ratio - original_ratio) < 0.1
        assert abs(test_ratio - original_ratio) < 0.1

    def test_split_data_reproducible(self, sample_data, temp_data_dir):
        """Test splitting is reproducible with same random state."""
        loader = DataLoader(temp_data_dir)

        train1, test1 = loader.split_data(sample_data, random_state=42)
        train2, test2 = loader.split_data(sample_data, random_state=42)

        pd.testing.assert_frame_equal(train1.reset_index(drop=True),
                                      train2.reset_index(drop=True))

    def test_get_data_summary(self, sample_data, temp_data_dir):
        """Test data summary generation."""
        loader = DataLoader(temp_data_dir)
        summary = loader.get_data_summary(sample_data)

        assert summary['shape']['rows'] == 100
        assert 'age_months' in summary['numeric_summary']
        assert 'gender' in summary['categorical_summary']
        assert 'target_distribution' in summary

    def test_save_processed_data(self, sample_data, temp_data_dir):
        """Test saving processed data."""
        loader = DataLoader(temp_data_dir)
        output_path = loader.save_processed_data(
            sample_data,
            'processed_data.csv'
        )

        assert output_path.exists()
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == len(sample_data)


class TestLoadConfig:
    """Test cases for config loading function."""

    def test_load_valid_config(self):
        """Test loading valid YAML config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
data_processing:
  test_size: 0.2
  random_state: 42
model:
  max_depth: 6
""")
            f.flush()

            config = load_config(f.name)

            assert config['data_processing']['test_size'] == 0.2
            assert config['model']['max_depth'] == 6

        os.unlink(f.name)

    def test_load_nonexistent_config(self):
        """Test loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.yaml')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
