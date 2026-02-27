"""
Pytest Configuration and Shared Fixtures

Provides common fixtures and configuration for all tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope='session')
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope='session')
def data_dir(project_root):
    """Return data directory."""
    return project_root / 'data'


@pytest.fixture(scope='session')
def sample_asd_data():
    """Create comprehensive sample ASD screening data."""
    np.random.seed(42)
    n_samples = 300

    # Generate ASD labels (30% prevalence for training)
    asd_labels = np.random.binomial(1, 0.30, n_samples)

    data = {
        'participant_id': [f'P{i:05d}' for i in range(n_samples)],
        'age_months': np.random.randint(18, 37, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
        'geographic_location': np.random.choice(
            ['Harare', 'Bulawayo', 'Manicaland', 'Masvingo'],
            n_samples
        ),
        'setting_type': np.random.choice(['urban', 'peri-urban', 'rural'], n_samples),

        # Behavioral features (correlated with ASD)
        'eye_contact': np.where(asd_labels, np.random.binomial(1, 0.7, n_samples),
                                np.random.binomial(1, 0.15, n_samples)),
        'response_to_name': np.where(asd_labels, np.random.binomial(1, 0.65, n_samples),
                                     np.random.binomial(1, 0.1, n_samples)),
        'pointing': np.where(asd_labels, np.random.binomial(1, 0.7, n_samples),
                            np.random.binomial(1, 0.12, n_samples)),
        'social_smile': np.where(asd_labels, np.random.binomial(1, 0.55, n_samples),
                                np.random.binomial(1, 0.08, n_samples)),
        'repetitive_behaviors': np.where(asd_labels, np.random.binomial(1, 0.72, n_samples),
                                         np.random.binomial(1, 0.1, n_samples)),
        'joint_attention': np.where(asd_labels, np.random.binomial(1, 0.68, n_samples),
                                   np.random.binomial(1, 0.12, n_samples)),

        # Communication features
        'word_count': np.where(asd_labels,
                              np.clip(np.random.normal(40, 30, n_samples), 0, 200),
                              np.clip(np.random.normal(150, 50, n_samples), 20, 350)).astype(int),
        'two_word_phrases': np.where(asd_labels, np.random.binomial(1, 0.25, n_samples),
                                    np.random.binomial(1, 0.85, n_samples)),
        'echolalia': np.where(asd_labels, np.random.binomial(1, 0.55, n_samples),
                             np.random.binomial(1, 0.1, n_samples)),
        'language_regression': np.where(asd_labels, np.random.binomial(1, 0.3, n_samples),
                                       np.random.binomial(1, 0.02, n_samples)),

        # Screening scores
        'mchat_score': np.where(asd_labels,
                               np.clip(np.random.normal(12, 3, n_samples), 5, 20),
                               np.clip(np.random.normal(2, 1.5, n_samples), 0, 8)).astype(int),

        # Demographics
        'family_history_asd': np.where(asd_labels, np.random.binomial(1, 0.2, n_samples),
                                      np.random.binomial(1, 0.05, n_samples)),
        'gestational_weeks': np.clip(np.random.normal(39, 2, n_samples), 28, 42).astype(int),

        # Target
        'asd_diagnosis': asd_labels
    }

    return pd.DataFrame(data)


@pytest.fixture
def small_sample_data():
    """Create small sample data for quick tests."""
    np.random.seed(42)
    n_samples = 50

    asd_labels = np.random.binomial(1, 0.3, n_samples)

    data = {
        'age_months': np.random.randint(18, 37, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'eye_contact': np.where(asd_labels, np.random.binomial(1, 0.7, n_samples),
                                np.random.binomial(1, 0.15, n_samples)),
        'response_to_name': np.where(asd_labels, np.random.binomial(1, 0.65, n_samples),
                                     np.random.binomial(1, 0.1, n_samples)),
        'mchat_score': np.where(asd_labels,
                               np.random.randint(8, 20, n_samples),
                               np.random.randint(0, 8, n_samples)),
        'asd_diagnosis': asd_labels
    }

    return pd.DataFrame(data)


@pytest.fixture
def train_test_split_data(sample_asd_data):
    """Split sample data into train and test sets."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        sample_asd_data,
        test_size=0.2,
        random_state=42,
        stratify=sample_asd_data['asd_diagnosis']
    )

    return train_df, test_df


@pytest.fixture
def numeric_features_and_target(sample_asd_data):
    """Extract numeric features and target."""
    target = sample_asd_data['asd_diagnosis']

    # Select only numeric columns, excluding target and ID
    numeric_cols = sample_asd_data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['asd_diagnosis', 'participant_id']]

    features = sample_asd_data[numeric_cols]

    return features, target


# Markers for test categories
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
