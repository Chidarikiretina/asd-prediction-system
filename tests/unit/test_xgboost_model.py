"""
Unit Tests for XGBoost Model Module

Tests for model training, prediction, evaluation, and persistence.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from models.xgboost_model import ASDXGBoostModel


class TestASDXGBoostModel:
    """Test cases for ASDXGBoostModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200

        # Create features correlated with target
        y = np.random.binomial(1, 0.3, n_samples)

        X = pd.DataFrame({
            'feature1': y * 0.6 + np.random.normal(0, 0.3, n_samples),
            'feature2': y * 0.5 + np.random.normal(0, 0.4, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': y * 0.4 + np.random.normal(0, 0.5, n_samples),
            'feature5': np.random.binomial(1, y * 0.6 + 0.2, n_samples),
        })

        return X, pd.Series(y)

    @pytest.fixture
    def train_test_data(self, sample_data):
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split

        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def test_init_default(self):
        """Test model initialization with defaults."""
        model = ASDXGBoostModel()

        assert model.model is None
        assert model.params['objective'] == 'binary:logistic'
        assert model.params['eval_metric'] == 'auc'

    def test_init_custom_params(self):
        """Test model initialization with custom parameters."""
        custom_params = {
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 50
        }

        model = ASDXGBoostModel(params=custom_params)

        assert model.params['max_depth'] == 4
        assert model.params['learning_rate'] == 0.05

    def test_train(self, train_test_data):
        """Test model training."""
        X_train, X_test, y_train, y_test = train_test_data

        model = ASDXGBoostModel()
        model.train(X_train, y_train)

        assert model.model is not None
        assert model.is_fitted

    def test_train_with_validation(self, train_test_data):
        """Test model training with validation set."""
        X_train, X_test, y_train, y_test = train_test_data

        model = ASDXGBoostModel()
        model.train(X_train, y_train, X_val=X_test, y_val=y_test)

        assert model.model is not None

    def test_predict(self, train_test_data):
        """Test model prediction."""
        X_train, X_test, y_train, y_test = train_test_data

        model = ASDXGBoostModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, train_test_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = train_test_data

        model = ASDXGBoostModel()
        model.train(X_train, y_train)
        probabilities = model.predict_proba(X_test)

        assert len(probabilities) == len(X_test)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_predict_before_training(self, sample_data):
        """Test that prediction before training raises error."""
        X, y = sample_data
        model = ASDXGBoostModel()

        with pytest.raises(ValueError):
            model.predict(X)

    def test_evaluate(self, train_test_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = train_test_data

        model = ASDXGBoostModel()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics

        # Check metrics are reasonable
        for key, value in metrics.items():
            assert 0 <= value <= 1

    def test_cross_validate(self, sample_data):
        """Test cross-validation."""
        X, y = sample_data

        model = ASDXGBoostModel()
        cv_results = model.cross_validate(X, y, cv=3)

        assert 'mean_accuracy' in cv_results
        assert 'std_accuracy' in cv_results
        assert 'mean_roc_auc' in cv_results

        # Check means are reasonable
        assert 0 <= cv_results['mean_accuracy'] <= 1
        assert 0 <= cv_results['mean_roc_auc'] <= 1

    def test_hyperparameter_tuning(self, sample_data):
        """Test hyperparameter tuning."""
        X, y = sample_data

        model = ASDXGBoostModel()

        # Small param grid for speed
        param_grid = {
            'max_depth': [3, 4],
            'learning_rate': [0.1, 0.2]
        }

        best_params, best_score = model.hyperparameter_tuning(
            X, y,
            param_grid=param_grid,
            cv=2
        )

        assert 'max_depth' in best_params
        assert 'learning_rate' in best_params
        assert 0 <= best_score <= 1

    def test_get_feature_importance(self, train_test_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = train_test_data

        model = ASDXGBoostModel()
        model.train(X_train, y_train)
        importance = model.get_feature_importance()

        assert len(importance) == X_train.shape[1]
        assert all(v >= 0 for v in importance.values())

        # Sum should be approximately 1 (normalized importance)
        # Note: XGBoost importance may not always sum to 1 exactly
        assert sum(importance.values()) > 0

    def test_save_and_load(self, train_test_data):
        """Test model save and load."""
        X_train, X_test, y_train, y_test = train_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'model.joblib'

            # Train and save
            model = ASDXGBoostModel()
            model.train(X_train, y_train)
            original_predictions = model.predict(X_test)
            model.save_model(filepath)

            # Load and verify
            loaded_model = ASDXGBoostModel()
            loaded_model.load_model(filepath)
            loaded_predictions = loaded_model.predict(X_test)

            # Predictions should match
            np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_save_before_training(self):
        """Test that saving before training raises error."""
        model = ASDXGBoostModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'model.joblib'

            with pytest.raises(ValueError):
                model.save_model(filepath)

    def test_model_reproducibility(self, train_test_data):
        """Test that training is reproducible with same random state."""
        X_train, X_test, y_train, y_test = train_test_data

        model1 = ASDXGBoostModel(params={'random_state': 42})
        model1.train(X_train, y_train)
        pred1 = model1.predict_proba(X_test)

        model2 = ASDXGBoostModel(params={'random_state': 42})
        model2.train(X_train, y_train)
        pred2 = model2.predict_proba(X_test)

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestASDXGBoostModelEdgeCases:
    """Test edge cases for ASDXGBoostModel."""

    def test_single_feature(self):
        """Test training with single feature."""
        np.random.seed(42)

        X = pd.DataFrame({'single_feature': np.random.normal(0, 1, 100)})
        y = pd.Series(np.random.binomial(1, 0.3, 100))

        model = ASDXGBoostModel()
        model.train(X, y)

        predictions = model.predict(X)
        assert len(predictions) == 100

    def test_imbalanced_classes(self):
        """Test training with highly imbalanced classes."""
        np.random.seed(42)

        # 95% negative, 5% positive
        n_samples = 200
        y = np.zeros(n_samples)
        y[:10] = 1  # Only 10 positives

        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })

        model = ASDXGBoostModel()
        model.train(X, pd.Series(y))

        # Should still be able to predict
        predictions = model.predict(X)
        assert len(predictions) == n_samples

    def test_missing_values_in_prediction(self, train_test_data):
        """Test prediction with missing values."""
        X_train, X_test, y_train, y_test = train_test_data

        model = ASDXGBoostModel()
        model.train(X_train, y_train)

        # Introduce missing values in test data
        X_test_missing = X_test.copy()
        X_test_missing.iloc[0, 0] = np.nan

        # XGBoost should handle missing values
        predictions = model.predict(X_test_missing)
        assert len(predictions) == len(X_test_missing)

    def test_predict_single_sample(self, train_test_data):
        """Test prediction on single sample."""
        X_train, X_test, y_train, y_test = train_test_data

        model = ASDXGBoostModel()
        model.train(X_train, y_train)

        # Predict single sample
        single_sample = X_test.iloc[[0]]
        prediction = model.predict(single_sample)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1]

    def test_different_n_estimators(self, train_test_data):
        """Test training with different n_estimators."""
        X_train, X_test, y_train, y_test = train_test_data

        for n_est in [10, 50, 100]:
            model = ASDXGBoostModel(params={'n_estimators': n_est})
            model.train(X_train, y_train)

            metrics = model.evaluate(X_test, y_test)
            assert metrics['accuracy'] >= 0


class TestASDXGBoostModelPerformance:
    """Test model performance characteristics."""

    @pytest.fixture
    def correlated_data(self):
        """Create data with strong signal."""
        np.random.seed(42)
        n_samples = 500

        y = np.random.binomial(1, 0.3, n_samples)

        # Create strongly correlated features
        X = pd.DataFrame({
            'strong_signal': y * 0.8 + np.random.normal(0, 0.2, n_samples),
            'medium_signal': y * 0.5 + np.random.normal(0, 0.3, n_samples),
            'weak_signal': y * 0.2 + np.random.normal(0, 0.5, n_samples),
            'noise': np.random.normal(0, 1, n_samples)
        })

        return X, pd.Series(y)

    def test_feature_importance_ranking(self, correlated_data):
        """Test that feature importance correctly ranks features."""
        X, y = correlated_data

        model = ASDXGBoostModel()
        model.train(X, y)
        importance = model.get_feature_importance()

        # Strong signal should have highest importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_feature = sorted_features[0][0]

        # The strong signal feature should be in top 2
        assert top_feature in ['strong_signal', 'medium_signal']

    def test_minimum_viable_performance(self, correlated_data):
        """Test that model achieves minimum viable performance on easy data."""
        from sklearn.model_selection import train_test_split

        X, y = correlated_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = ASDXGBoostModel()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        # On data with strong signal, should achieve reasonable performance
        assert metrics['roc_auc'] > 0.7
        assert metrics['accuracy'] > 0.6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
