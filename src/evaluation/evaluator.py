"""
Model Evaluation Module

Comprehensive evaluation utilities for ASD prediction models including
metrics calculation, visualization, and reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for ASD prediction.

    Provides metrics calculation, threshold optimization,
    visualization, and clinical performance assessment.
    """

    # Target performance metrics for ASD screening
    TARGET_METRICS = {
        'sensitivity': 0.90,  # Recall - critical for screening
        'specificity': 0.85,
        'roc_auc': 0.92
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.results: Dict[str, Any] = {}
        self.threshold = 0.5

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

        # Additional metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Prevalence and rates
        metrics['prevalence'] = y_true.mean()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Probability-based metrics (if available)
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['average_precision'] = average_precision_score(y_true, y_prob)

            # Brier score (lower is better)
            metrics['brier_score'] = np.mean((y_prob - y_true) ** 2)

        # Confusion matrix counts
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)

        self.results['metrics'] = metrics
        return metrics

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric: str = 'f1_score',
        min_sensitivity: float = 0.85
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal classification threshold.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            metric: Metric to optimize ('f1_score', 'balanced_accuracy', 'youden_j')
            min_sensitivity: Minimum required sensitivity

        Returns:
            Tuple of (optimal threshold, metrics at threshold)
        """
        logger.info(f"Optimizing threshold for {metric} with min sensitivity {min_sensitivity}")

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        best_metrics = {}

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            metrics = self.calculate_metrics(y_true, y_pred, y_prob)

            # Check sensitivity constraint
            if metrics['sensitivity'] < min_sensitivity:
                continue

            if metric == 'f1_score':
                score = metrics['f1_score']
            elif metric == 'balanced_accuracy':
                score = metrics['balanced_accuracy']
            elif metric == 'youden_j':
                score = metrics['sensitivity'] + metrics['specificity'] - 1
            else:
                score = metrics.get(metric, 0)

            if score > best_score:
                best_score = score
                best_threshold = thresh
                best_metrics = metrics.copy()

        self.threshold = best_threshold
        self.results['optimal_threshold'] = best_threshold
        self.results['threshold_metrics'] = best_metrics

        logger.info(f"Optimal threshold: {best_threshold:.3f} with {metric}={best_score:.4f}")
        return best_threshold, best_metrics

    def evaluate_clinical_utility(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        population_size: int = 10000
    ) -> Dict[str, Any]:
        """
        Evaluate clinical utility metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            population_size: Hypothetical screening population size

        Returns:
            Dictionary with clinical utility metrics
        """
        metrics = self.calculate_metrics(y_true, y_pred)

        prevalence = metrics['prevalence']
        sensitivity = metrics['sensitivity']
        specificity = metrics['specificity']

        # Expected outcomes in population
        expected_cases = int(population_size * prevalence)
        expected_non_cases = population_size - expected_cases

        detected_cases = int(expected_cases * sensitivity)
        missed_cases = expected_cases - detected_cases
        false_alarms = int(expected_non_cases * (1 - specificity))
        true_negatives = expected_non_cases - false_alarms

        # Number needed to screen
        nns = int(1 / (prevalence * sensitivity)) if (prevalence * sensitivity) > 0 else float('inf')

        clinical_metrics = {
            'population_screened': population_size,
            'expected_cases': expected_cases,
            'cases_detected': detected_cases,
            'cases_missed': missed_cases,
            'false_alarms': false_alarms,
            'correctly_cleared': true_negatives,
            'number_needed_to_screen': nns,
            'detection_rate': detected_cases / expected_cases if expected_cases > 0 else 0,
            'referral_rate': (detected_cases + false_alarms) / population_size
        }

        self.results['clinical_utility'] = clinical_metrics
        return clinical_metrics

    def check_target_performance(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if metrics meet target performance criteria.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Dictionary indicating which targets are met
        """
        targets_met = {}

        for metric_name, target_value in self.TARGET_METRICS.items():
            if metric_name in metrics:
                targets_met[metric_name] = metrics[metric_name] >= target_value
                status = "✓" if targets_met[metric_name] else "✗"
                logger.info(f"{metric_name}: {metrics[metric_name]:.4f} (target: {target_value}) {status}")

        self.results['targets_met'] = targets_met
        return targets_met

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None,
        normalize: bool = False
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save figure
            normalize: Whether to normalize the matrix

        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2%' if normalize else 'd',
            cmap='Blues',
            xticklabels=['No ASD', 'ASD'],
            yticklabels=['No ASD', 'ASD'],
            ax=ax
        )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

        # Mark target sensitivity/specificity point
        target_sens = self.TARGET_METRICS.get('sensitivity', 0.9)
        target_spec = self.TARGET_METRICS.get('specificity', 0.85)
        ax.scatter([1-target_spec], [target_sens], color='red', s=100, zorder=5,
                   label=f'Target (Sens={target_sens}, Spec={target_spec})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')

        # Baseline (random classifier)
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='navy', linestyle='--',
                   label=f'Baseline (prevalence = {baseline:.2f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (PPV)')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")

        return fig

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot calibration curve.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            n_bins: Number of bins for calibration
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Calibration curve
        ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curve')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Histogram of predictions
        ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.5, label='No ASD', density=True)
        ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.5, label='ASD', density=True)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {save_path}")

        return fig

    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot metrics across different thresholds.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        thresholds = np.arange(0.1, 0.9, 0.02)
        metrics_by_thresh = {
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'f1_score': []
        }

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            metrics = self.calculate_metrics(y_true, y_pred)
            for metric_name in metrics_by_thresh:
                metrics_by_thresh[metric_name].append(metrics[metric_name])

        fig, ax = plt.subplots(figsize=(10, 6))

        for metric_name, values in metrics_by_thresh.items():
            ax.plot(thresholds, values, lw=2, label=metric_name.replace('_', ' ').title())

        # Mark target thresholds
        ax.axhline(y=self.TARGET_METRICS.get('sensitivity', 0.9), color='red',
                   linestyle=':', alpha=0.7, label='Target Sensitivity')
        ax.axhline(y=self.TARGET_METRICS.get('specificity', 0.85), color='blue',
                   linestyle=':', alpha=0.7, label='Target Specificity')

        ax.set_xlabel('Classification Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title('Metrics vs. Classification Threshold')
        ax.legend(loc='center right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.1, 0.9])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Threshold analysis saved to {save_path}")

        return fig

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None,
        model_name: str = 'ASD_Prediction_Model'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            output_dir: Directory to save report and figures
            model_name: Name of the model for the report

        Returns:
            Complete evaluation report dictionary
        """
        logger.info("Generating comprehensive evaluation report")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate all metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)

        # Check target performance
        targets_met = self.check_target_performance(metrics)

        # Clinical utility
        clinical = self.evaluate_clinical_utility(y_true, y_pred)

        # Optimize threshold if probabilities available
        if y_prob is not None:
            optimal_thresh, thresh_metrics = self.optimize_threshold(
                y_true, y_prob, metric='f1_score', min_sensitivity=0.85
            )

        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=['No ASD', 'ASD'])

        # Compile report
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(y_true),
                'positive_samples': int(y_true.sum()),
                'negative_samples': int(len(y_true) - y_true.sum()),
                'prevalence': float(y_true.mean())
            },
            'metrics': metrics,
            'targets_met': targets_met,
            'all_targets_met': all(targets_met.values()),
            'clinical_utility': clinical,
            'classification_report': class_report
        }

        if y_prob is not None:
            report['optimal_threshold'] = optimal_thresh
            report['metrics_at_optimal_threshold'] = thresh_metrics

        # Generate and save plots
        if output_dir:
            self.plot_confusion_matrix(y_true, y_pred,
                                      save_path=output_dir / 'confusion_matrix.png')
            self.plot_confusion_matrix(y_true, y_pred, normalize=True,
                                      save_path=output_dir / 'confusion_matrix_normalized.png')

            if y_prob is not None:
                self.plot_roc_curve(y_true, y_prob,
                                   save_path=output_dir / 'roc_curve.png')
                self.plot_precision_recall_curve(y_true, y_prob,
                                                save_path=output_dir / 'pr_curve.png')
                self.plot_calibration_curve(y_true, y_prob,
                                           save_path=output_dir / 'calibration_curve.png')
                self.plot_threshold_analysis(y_true, y_prob,
                                            save_path=output_dir / 'threshold_analysis.png')

            # Save JSON report
            report_path = output_dir / 'evaluation_report.json'

            # Convert numpy types for JSON serialization
            report_json = self._convert_to_json_serializable(report)
            with open(report_path, 'w') as f:
                json.dump(report_json, f, indent=2)
            logger.info(f"Report saved to {report_path}")

        self.results = report
        return report

    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    def print_summary(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Print a formatted summary of evaluation results.

        Args:
            metrics: Metrics dictionary (uses stored results if None)
        """
        if metrics is None:
            metrics = self.results.get('metrics', {})

        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)

        print("\nCore Performance Metrics:")
        print("-"*40)
        print(f"  Accuracy:          {metrics.get('accuracy', 0):.4f}")
        print(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        print(f"  Precision (PPV):   {metrics.get('precision', 0):.4f}")
        print(f"  Recall (Sens):     {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:          {metrics.get('f1_score', 0):.4f}")

        print("\nClinical Performance:")
        print("-"*40)
        print(f"  Sensitivity:       {metrics.get('sensitivity', 0):.4f}")
        print(f"  Specificity:       {metrics.get('specificity', 0):.4f}")
        print(f"  PPV:               {metrics.get('ppv', 0):.4f}")
        print(f"  NPV:               {metrics.get('npv', 0):.4f}")

        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:           {metrics.get('roc_auc', 0):.4f}")

        print("\nConfusion Matrix:")
        print("-"*40)
        print(f"  True Positives:    {metrics.get('true_positives', 0)}")
        print(f"  True Negatives:    {metrics.get('true_negatives', 0)}")
        print(f"  False Positives:   {metrics.get('false_positives', 0)}")
        print(f"  False Negatives:   {metrics.get('false_negatives', 0)}")

        print("\nTarget Performance Check:")
        print("-"*40)
        for metric_name, target in self.TARGET_METRICS.items():
            actual = metrics.get(metric_name, 0)
            status = "✓ MET" if actual >= target else "✗ NOT MET"
            print(f"  {metric_name.title()}: {actual:.4f} (target: {target}) {status}")

        print("="*60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    evaluator = ModelEvaluator()

    # Example with synthetic data
    # np.random.seed(42)
    # y_true = np.random.binomial(1, 0.3, 200)
    # y_prob = np.clip(y_true * 0.6 + np.random.normal(0.3, 0.2, 200), 0, 1)
    # y_pred = (y_prob > 0.5).astype(int)
    #
    # report = evaluator.generate_report(
    #     y_true, y_pred, y_prob,
    #     output_dir=Path('outputs/reports'),
    #     model_name='Test_Model'
    # )
    # evaluator.print_summary()
