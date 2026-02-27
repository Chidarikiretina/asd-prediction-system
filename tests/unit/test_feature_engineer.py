"""
Unit Tests for Feature Engineering Module

Tests for feature creation, selection, and transformation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from feature_engineering.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200

        data = {
            'age_months': np.random.randint(18, 37, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'eye_contact': np.random.randint(0, 2, n_samples),
            'response_to_name': np.random.randint(0, 2, n_samples),
            'pointing': np.random.randint(0, 2, n_samples),
            'social_smile': np.random.randint(0, 2, n_samples),
            'repetitive_behaviors': np.random.randint(0, 2, n_samples),
            'joint_attention': np.random.randint(0, 2, n_samples),
            'pretend_play': np.random.randint(0, 2, n_samples),
            'hand_flapping': np.random.randint(0, 2, n_samples),
            'word_count': np.random.randint(0, 300, n_samples),
            'two_word_phrases': np.random.randint(0, 2, n_samples),
            'echolalia': np.random.randint(0, 2, n_samples),
            'language_regression': np.random.randint(0, 2, n_samples),
            'mchat_score': np.random.randint(0, 20, n_samples),
            'social_communication_score': np.random.uniform(0, 10, n_samples),
            'rrb_score': np.random.uniform(0, 10, n_samples),
            'gestational_weeks': np.random.randint(32, 42, n_samples),
            'family_history_asd': np.random.randint(0, 2, n_samples),
            'asd_diagnosis': np.random.randint(0, 2, n_samples)
        }

        return pd.DataFrame(data)

    def test_init(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert engineer.selected_features == []
        assert engineer.feature_importances == {}
        assert engineer._fitted is False

    def test_create_behavioral_composite_scores(self, sample_data):
        """Test behavioral composite score creation."""
        engineer = FeatureEngineer()
        df = engineer.create_behavioral_composite_scores(sample_data)

        assert 'behavioral_concern_total' in df.columns
        assert 'behavioral_concern_pct' in df.columns
        assert 'social_interaction_score' in df.columns

        # Check values are reasonable
        assert df['behavioral_concern_total'].min() >= 0
        assert df['behavioral_concern_pct'].max() <= 1.0

    def test_create_communication_features(self, sample_data):
        """Test communication feature creation."""
        engineer = FeatureEngineer()
        df = engineer.create_communication_features(sample_data)

        assert 'word_count_category' in df.columns
        assert 'communication_concern_score' in df.columns

        # Check word count categories
        valid_categories = ['none', 'minimal', 'delayed', 'typical']
        assert all(df['word_count_category'].dropna().isin(valid_categories))

    def test_create_demographic_interactions(self, sample_data):
        """Test demographic interaction creation."""
        engineer = FeatureEngineer()

        # First create behavioral scores needed for interactions
        df = engineer.create_behavioral_composite_scores(sample_data)
        df = engineer.create_demographic_interactions(df)

        assert 'age_group' in df.columns
        assert 'is_male' in df.columns
        assert 'preterm' in df.columns

        # Check age groups
        valid_age_groups = ['18-24m', '25-30m', '31-36m']
        assert all(df['age_group'].dropna().isin(valid_age_groups))

    def test_create_screening_score_features(self, sample_data):
        """Test screening score feature creation."""
        engineer = FeatureEngineer()
        df = engineer.create_screening_score_features(sample_data)

        assert 'mchat_risk_level' in df.columns
        assert 'mchat_high_risk' in df.columns
        assert 'combined_screening_score' in df.columns

        # Check M-CHAT risk levels
        valid_levels = ['low', 'medium', 'high']
        assert all(df['mchat_risk_level'].dropna().isin(valid_levels))

    def test_create_risk_indicators(self, sample_data):
        """Test risk indicator creation."""
        engineer = FeatureEngineer()

        # First create required intermediate features
        df = engineer.create_behavioral_composite_scores(sample_data)
        df = engineer.create_screening_score_features(df)
        df = engineer.create_risk_indicators(df)

        assert 'total_risk_indicators' in df.columns
        assert df['total_risk_indicators'].min() >= 0

    def test_engineer_all_features(self, sample_data):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer()
        df = engineer.engineer_all_features(sample_data)

        # Check that new features were added
        assert len(df.columns) > len(sample_data.columns)

        # Check key composite features exist
        assert 'behavioral_concern_total' in df.columns
        assert 'mchat_risk_level' in df.columns

    def test_select_features_by_importance_random_forest(self, sample_data):
        """Test feature selection using random forest importance."""
        engineer = FeatureEngineer()

        # Prepare numeric features
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('asd_diagnosis')
        X = sample_data[numeric_cols]
        y = sample_data['asd_diagnosis']

        X_selected, importances = engineer.select_features_by_importance(
            X, y,
            method='random_forest',
            n_features=5
        )

        assert X_selected.shape[1] == 5
        assert len(importances) == len(numeric_cols)
        assert engineer._fitted is True
        assert len(engineer.selected_features) == 5

    def test_select_features_by_importance_mutual_info(self, sample_data):
        """Test feature selection using mutual information."""
        engineer = FeatureEngineer()

        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('asd_diagnosis')
        X = sample_data[numeric_cols]
        y = sample_data['asd_diagnosis']

        X_selected, importances = engineer.select_features_by_importance(
            X, y,
            method='mutual_info',
            n_features=5
        )

        assert X_selected.shape[1] == 5
        assert all(imp >= 0 for imp in importances.values())

    def test_select_features_by_importance_threshold(self, sample_data):
        """Test feature selection using importance threshold."""
        engineer = FeatureEngineer()

        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('asd_diagnosis')
        X = sample_data[numeric_cols]
        y = sample_data['asd_diagnosis']

        X_selected, importances = engineer.select_features_by_importance(
            X, y,
            method='random_forest',
            n_features=None,
            threshold=0.05
        )

        # Should select features with importance >= 0.05
        for feat in engineer.selected_features:
            assert importances[feat] >= 0.05

    def test_select_features_rfe(self, sample_data):
        """Test recursive feature elimination."""
        engineer = FeatureEngineer()

        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('asd_diagnosis')
        X = sample_data[numeric_cols]
        y = sample_data['asd_diagnosis']

        X_selected, selected_features = engineer.select_features_rfe(
            X, y,
            n_features=5
        )

        assert X_selected.shape[1] == 5
        assert len(selected_features) == 5
        assert engineer._fitted is True

    def test_remove_low_variance_features(self, sample_data):
        """Test removal of low variance features."""
        engineer = FeatureEngineer()

        # Add a constant column (zero variance)
        df = sample_data.copy()
        df['constant_col'] = 1

        df_filtered = engineer.remove_low_variance_features(df, threshold=0.01)

        # Constant column should be removed
        assert 'constant_col' not in df_filtered.columns

    def test_apply_pca(self, sample_data):
        """Test PCA dimensionality reduction."""
        engineer = FeatureEngineer()

        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        X = sample_data[numeric_cols]

        pca_df = engineer.apply_pca(X, n_components=5)

        assert pca_df.shape[1] == 5
        assert all(col.startswith('pca_') for col in pca_df.columns)
        assert engineer.pca is not None

    def test_apply_pca_variance_ratio(self, sample_data):
        """Test PCA with variance ratio."""
        engineer = FeatureEngineer()

        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        X = sample_data[numeric_cols]

        pca_df = engineer.apply_pca(X, n_components=0.95)

        # Should preserve 95% of variance
        assert engineer.pca.explained_variance_ratio_.sum() >= 0.95

    def test_get_feature_importance_report(self, sample_data):
        """Test feature importance report generation."""
        engineer = FeatureEngineer()

        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('asd_diagnosis')
        X = sample_data[numeric_cols]
        y = sample_data['asd_diagnosis']

        engineer.select_features_by_importance(X, y, method='random_forest')
        report = engineer.get_feature_importance_report()

        assert isinstance(report, pd.DataFrame)
        assert 'feature' in report.columns
        assert 'importance' in report.columns
        assert 'rank' in report.columns
        assert 'cumulative_importance' in report.columns

    def test_save_and_load(self, sample_data):
        """Test saving and loading feature engineer state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'feature_engineer.joblib'

            # Fit and save
            engineer = FeatureEngineer()
            numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols.remove('asd_diagnosis')
            X = sample_data[numeric_cols]
            y = sample_data['asd_diagnosis']

            engineer.select_features_by_importance(X, y, n_features=5)
            engineer.save(filepath)

            # Load and verify
            loaded = FeatureEngineer.load(filepath)

            assert loaded._fitted == engineer._fitted
            assert loaded.selected_features == engineer.selected_features
            assert loaded.feature_importances == engineer.feature_importances


class TestFeatureEngineerEdgeCases:
    """Test edge cases for FeatureEngineer."""

    def test_missing_behavioral_features(self):
        """Test handling when behavioral features are missing."""
        engineer = FeatureEngineer()

        df = pd.DataFrame({
            'age_months': [24, 30, 18],
            'gender': ['M', 'F', 'M'],
            'asd_diagnosis': [0, 1, 0]
        })

        # Should handle gracefully
        df_processed = engineer.create_behavioral_composite_scores(df)
        assert 'behavioral_concern_total' not in df_processed.columns

    def test_partial_behavioral_features(self):
        """Test handling when only some behavioral features exist."""
        engineer = FeatureEngineer()

        df = pd.DataFrame({
            'age_months': [24, 30, 18],
            'eye_contact': [1, 0, 1],
            'social_smile': [0, 1, 1],
            'asd_diagnosis': [0, 1, 0]
        })

        df_processed = engineer.create_behavioral_composite_scores(df)

        # Should create composite from available features
        assert 'behavioral_concern_total' in df_processed.columns
        assert df_processed['behavioral_concern_total'].max() <= 2

    def test_empty_feature_selection(self):
        """Test feature selection with empty result."""
        engineer = FeatureEngineer()

        df = pd.DataFrame({
            'feature1': [1, 1, 1, 1, 1],  # No variance
            'feature2': [0, 0, 0, 0, 0],  # No variance
            'target': [0, 1, 0, 1, 0]
        })

        X = df[['feature1', 'feature2']]
        y = df['target']

        # Should handle gracefully
        X_selected, importances = engineer.select_features_by_importance(
            X, y,
            method='random_forest',
            threshold=0.99  # Very high threshold
        )

        # May return empty or low-importance features
        assert isinstance(X_selected, pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
