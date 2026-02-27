"""
Unit Tests for Model Evaluator Module

Tests for evaluation metrics, visualization, and reporting.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from evaluation.evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""

    @pytest.fixture
    def binary_predictions(self):
        """Create sample binary predictions."""
        np.random.seed(42)
        n_samples = 200

        # Create realistic predictions
        y_true = np.random.binomial(1, 0.3, n_samples)

        # Create probabilities correlated with true labels
        y_prob = np.clip(
            y_true * 0.6 + np.random.normal(0.3, 0.2, n_samples),
            0, 1
        )
        y_pred = (y_prob > 0.5).astype(int)

        return y_true, y_pred, y_prob

    @pytest.fixture
    def perfect_predictions(self):
        """Create perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        return y_true, y_pred, y_prob

    @pytest.fixture
    def worst_predictions(self):
        """Create worst predictions (all wrong)."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        return y_true, y_pred, y_prob

    def test_init(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator.results == {}
        assert evaluator.threshold == 0.5

    def test_calculate_metrics_basic(self, binary_predictions):
        """Test basic metrics calculation."""
        y_true, y_pred, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)

        # Check all expected metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'sensitivity' in metrics
        assert 'specificity' in metrics
        assert 'roc_auc' in metrics

        # Check metrics are in valid ranges
        for key, value in metrics.items():
            if key not in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']:
                assert 0 <= value <= 1, f"{key} = {value} is out of range"

    def test_calculate_metrics_perfect(self, perfect_predictions):
        """Test metrics for perfect predictions."""
        y_true, y_pred, y_prob = perfect_predictions
        evaluator = ModelEvaluator()

        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)

        assert metrics['accuracy'] == 1.0
        assert metrics['sensitivity'] == 1.0
        assert metrics['specificity'] == 1.0
        assert metrics['f1_score'] == 1.0

    def test_calculate_metrics_worst(self, worst_predictions):
        """Test metrics for worst predictions."""
        y_true, y_pred, y_prob = worst_predictions
        evaluator = ModelEvaluator()

        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)

        assert metrics['accuracy'] == 0.0
        assert metrics['sensitivity'] == 0.0
        assert metrics['specificity'] == 0.0

    def test_calculate_metrics_without_probabilities(self, binary_predictions):
        """Test metrics calculation without probabilities."""
        y_true, y_pred, _ = binary_predictions
        evaluator = ModelEvaluator()

        metrics = evaluator.calculate_metrics(y_true, y_pred)

        # Basic metrics should still work
        assert 'accuracy' in metrics
        assert 'precision' in metrics

        # Probability-based metrics should not be present
        assert 'roc_auc' not in metrics or metrics.get('roc_auc') is None

    def test_confusion_matrix_counts(self, binary_predictions):
        """Test confusion matrix element counts."""
        y_true, y_pred, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)

        tp = metrics['true_positives']
        tn = metrics['true_negatives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']

        # Sum should equal total samples
        assert tp + tn + fp + fn == len(y_true)

        # Check positives match
        assert tp + fn == y_true.sum()

    def test_optimize_threshold(self, binary_predictions):
        """Test threshold optimization."""
        y_true, y_pred, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        optimal_thresh, metrics = evaluator.optimize_threshold(
            y_true, y_prob,
            metric='f1_score',
            min_sensitivity=0.5
        )

        assert 0 < optimal_thresh < 1
        assert metrics['sensitivity'] >= 0.5
        assert evaluator.threshold == optimal_thresh

    def test_optimize_threshold_balanced_accuracy(self, binary_predictions):
        """Test threshold optimization for balanced accuracy."""
        y_true, y_pred, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        optimal_thresh, metrics = evaluator.optimize_threshold(
            y_true, y_prob,
            metric='balanced_accuracy',
            min_sensitivity=0.5
        )

        assert 0 < optimal_thresh < 1
        assert 'balanced_accuracy' in metrics

    def test_optimize_threshold_youden_j(self, binary_predictions):
        """Test threshold optimization for Youden's J."""
        y_true, y_pred, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        optimal_thresh, metrics = evaluator.optimize_threshold(
            y_true, y_prob,
            metric='youden_j',
            min_sensitivity=0.5
        )

        assert 0 < optimal_thresh < 1

    def test_evaluate_clinical_utility(self, binary_predictions):
        """Test clinical utility evaluation."""
        y_true, y_pred, _ = binary_predictions
        evaluator = ModelEvaluator()

        clinical = evaluator.evaluate_clinical_utility(
            y_true, y_pred,
            population_size=10000
        )

        assert clinical['population_screened'] == 10000
        assert clinical['expected_cases'] > 0
        assert clinical['cases_detected'] >= 0
        assert clinical['number_needed_to_screen'] >= 1

        # Referral rate should be between 0 and 1
        assert 0 <= clinical['referral_rate'] <= 1

    def test_check_target_performance(self, binary_predictions):
        """Test target performance checking."""
        y_true, y_pred, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
        targets_met = evaluator.check_target_performance(metrics)

        # Should have entries for all target metrics
        for target_name in evaluator.TARGET_METRICS.keys():
            if target_name in metrics:
                assert target_name in targets_met

    def test_check_target_performance_perfect(self, perfect_predictions):
        """Test target performance with perfect predictions."""
        y_true, y_pred, y_prob = perfect_predictions
        evaluator = ModelEvaluator()

        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
        targets_met = evaluator.check_target_performance(metrics)

        # All targets should be met with perfect predictions
        assert targets_met.get('sensitivity', False) is True
        assert targets_met.get('specificity', False) is True

    def test_generate_report(self, binary_predictions):
        """Test report generation."""
        y_true, y_pred, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        with tempfile.TemporaryDirectory() as tmpdir:
            report = evaluator.generate_report(
                y_true, y_pred, y_prob,
                output_dir=Path(tmpdir),
                model_name='Test_Model'
            )

            # Check report structure
            assert report['model_name'] == 'Test_Model'
            assert 'timestamp' in report
            assert 'metrics' in report
            assert 'targets_met' in report
            assert 'clinical_utility' in report

            # Check files were created
            assert (Path(tmpdir) / 'confusion_matrix.png').exists()
            assert (Path(tmpdir) / 'roc_curve.png').exists()
            assert (Path(tmpdir) / 'evaluation_report.json').exists()

    def test_generate_report_without_output(self, binary_predictions):
        """Test report generation without saving files."""
        y_true, y_pred, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        report = evaluator.generate_report(
            y_true, y_pred, y_prob,
            output_dir=None,
            model_name='Test_Model'
        )

        assert 'metrics' in report
        assert report['dataset_info']['total_samples'] == len(y_true)


class TestModelEvaluatorPlots:
    """Test plotting functionality."""

    @pytest.fixture
    def binary_predictions(self):
        """Create sample binary predictions."""
        np.random.seed(42)
        n_samples = 100

        y_true = np.random.binomial(1, 0.3, n_samples)
        y_prob = np.clip(
            y_true * 0.6 + np.random.normal(0.3, 0.2, n_samples),
            0, 1
        )
        y_pred = (y_prob > 0.5).astype(int)

        return y_true, y_pred, y_prob

    def test_plot_confusion_matrix(self, binary_predictions):
        """Test confusion matrix plotting."""
        y_true, y_pred, _ = binary_predictions
        evaluator = ModelEvaluator()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig = evaluator.plot_confusion_matrix(
                y_true, y_pred,
                save_path=Path(tmpdir) / 'cm.png'
            )

            assert fig is not None
            assert (Path(tmpdir) / 'cm.png').exists()

    def test_plot_confusion_matrix_normalized(self, binary_predictions):
        """Test normalized confusion matrix plotting."""
        y_true, y_pred, _ = binary_predictions
        evaluator = ModelEvaluator()

        fig = evaluator.plot_confusion_matrix(y_true, y_pred, normalize=True)
        assert fig is not None

    def test_plot_roc_curve(self, binary_predictions):
        """Test ROC curve plotting."""
        y_true, _, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig = evaluator.plot_roc_curve(
                y_true, y_prob,
                save_path=Path(tmpdir) / 'roc.png'
            )

            assert fig is not None
            assert (Path(tmpdir) / 'roc.png').exists()

    def test_plot_precision_recall_curve(self, binary_predictions):
        """Test precision-recall curve plotting."""
        y_true, _, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig = evaluator.plot_precision_recall_curve(
                y_true, y_prob,
                save_path=Path(tmpdir) / 'pr.png'
            )

            assert fig is not None
            assert (Path(tmpdir) / 'pr.png').exists()

    def test_plot_calibration_curve(self, binary_predictions):
        """Test calibration curve plotting."""
        y_true, _, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        fig = evaluator.plot_calibration_curve(y_true, y_prob, n_bins=5)
        assert fig is not None

    def test_plot_threshold_analysis(self, binary_predictions):
        """Test threshold analysis plotting."""
        y_true, _, y_prob = binary_predictions
        evaluator = ModelEvaluator()

        fig = evaluator.plot_threshold_analysis(y_true, y_prob)
        assert fig is not None


class TestModelEvaluatorEdgeCases:
    """Test edge cases for ModelEvaluator."""

    def test_all_positive_predictions(self):
        """Test when all predictions are positive."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred)

        assert metrics['recall'] == 1.0  # All positives caught
        assert metrics['specificity'] == 0.0  # No negatives caught

    def test_all_negative_predictions(self):
        """Test when all predictions are negative."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred)

        assert metrics['recall'] == 0.0  # No positives caught
        assert metrics['specificity'] == 1.0  # All negatives caught

    def test_single_class_true(self):
        """Test when true labels have only one class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.4, 0.7])

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)

        # Should handle gracefully
        assert 'accuracy' in metrics

    def test_small_sample_size(self):
        """Test with very small sample size."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_prob = np.array([0.3, 0.8])

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)

        assert metrics['accuracy'] == 1.0

    def test_print_summary(self, capsys):
        """Test summary printing."""
        evaluator = ModelEvaluator()
        metrics = {
            'accuracy': 0.85,
            'balanced_accuracy': 0.83,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'sensitivity': 0.75,
            'specificity': 0.90,
            'ppv': 0.80,
            'npv': 0.88,
            'roc_auc': 0.88,
            'true_positives': 30,
            'true_negatives': 55,
            'false_positives': 8,
            'false_negatives': 7
        }

        evaluator.print_summary(metrics)
        captured = capsys.readouterr()

        assert 'MODEL EVALUATION SUMMARY' in captured.out
        assert 'Accuracy' in captured.out
        assert 'Sensitivity' in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
