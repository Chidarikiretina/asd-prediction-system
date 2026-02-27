"""
Feature Engineering Module

This module handles feature creation, selection, and transformation
for the ASD prediction system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for ASD prediction.

    Handles feature creation, selection, and importance analysis
    tailored for behavioral screening data.
    """

    # Behavioral feature groups for domain-specific engineering
    BEHAVIORAL_FEATURES = [
        'eye_contact', 'response_to_name', 'pointing', 'social_smile',
        'repetitive_behaviors', 'joint_attention', 'pretend_play',
        'unusual_interests', 'hand_flapping', 'toe_walking',
        'lines_up_toys', 'upset_by_change'
    ]

    COMMUNICATION_FEATURES = [
        'word_count', 'two_word_phrases', 'echolalia', 'language_regression'
    ]

    SCREENING_SCORES = [
        'mchat_score', 'social_communication_score', 'rrb_score'
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.selected_features: List[str] = []
        self.feature_importances: Dict[str, float] = {}
        self.selector = None
        self.pca = None
        self._fitted = False

    def create_behavioral_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite scores from behavioral features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new composite features
        """
        df = df.copy()
        logger.info("Creating behavioral composite scores")

        # Find available behavioral features
        available_behavioral = [f for f in self.BEHAVIORAL_FEATURES if f in df.columns]

        if available_behavioral:
            # Total behavioral concern score
            df['behavioral_concern_total'] = df[available_behavioral].sum(axis=1)

            # Behavioral concern percentage
            df['behavioral_concern_pct'] = (
                df[available_behavioral].sum(axis=1) / len(available_behavioral)
            )

            # Social interaction subset
            social_features = ['eye_contact', 'social_smile', 'joint_attention', 'response_to_name']
            available_social = [f for f in social_features if f in df.columns]
            if available_social:
                df['social_interaction_score'] = df[available_social].sum(axis=1)

            # Repetitive behavior subset
            rrb_features = ['repetitive_behaviors', 'hand_flapping', 'toe_walking', 'lines_up_toys']
            available_rrb = [f for f in rrb_features if f in df.columns]
            if available_rrb:
                df['rrb_behavioral_score'] = df[available_rrb].sum(axis=1)

            logger.info(f"Created composite scores from {len(available_behavioral)} behavioral features")

        return df

    def create_communication_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived communication features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new communication features
        """
        df = df.copy()
        logger.info("Creating communication features")

        # Word count categories
        if 'word_count' in df.columns:
            df['word_count_category'] = pd.cut(
                df['word_count'],
                bins=[-1, 10, 50, 150, 500],
                labels=['none', 'minimal', 'delayed', 'typical']
            )

            # Language delay indicator
            if 'age_months' in df.columns:
                # Expected words by age (rough approximation)
                expected_words = df['age_months'] * 5  # Simplified expectation
                df['language_delay_ratio'] = df['word_count'] / (expected_words + 1)
                df['has_language_delay'] = (df['language_delay_ratio'] < 0.5).astype(int)

        # Communication concern composite
        available_comm = [f for f in self.COMMUNICATION_FEATURES if f in df.columns]
        if available_comm:
            # Normalize and combine (some are binary, some continuous)
            comm_binary = ['two_word_phrases', 'echolalia', 'language_regression']
            available_binary = [f for f in comm_binary if f in df.columns]
            if available_binary:
                df['communication_concern_score'] = df[available_binary].sum(axis=1)

        return df

    def create_demographic_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic interaction features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with demographic interactions
        """
        df = df.copy()
        logger.info("Creating demographic interaction features")

        if 'age_months' in df.columns:
            # Age groups
            df['age_group'] = pd.cut(
                df['age_months'],
                bins=[17, 24, 30, 37],
                labels=['18-24m', '25-30m', '31-36m']
            )

            # Age-behavioral interactions
            if 'behavioral_concern_total' in df.columns:
                df['age_behavioral_interaction'] = df['age_months'] * df['behavioral_concern_total']

        # Gender encoding (if not already encoded)
        if 'gender' in df.columns and df['gender'].dtype == 'object':
            df['is_male'] = (df['gender'] == 'M').astype(int)

        # Gestational age feature
        if 'gestational_weeks' in df.columns:
            df['preterm'] = (df['gestational_weeks'] < 37).astype(int)
            df['very_preterm'] = (df['gestational_weeks'] < 32).astype(int)

        return df

    def create_screening_score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from screening scores.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with screening score features
        """
        df = df.copy()
        logger.info("Creating screening score features")

        if 'mchat_score' in df.columns:
            # M-CHAT risk categories
            df['mchat_risk_level'] = pd.cut(
                df['mchat_score'],
                bins=[-1, 2, 7, 20],
                labels=['low', 'medium', 'high']
            )
            df['mchat_high_risk'] = (df['mchat_score'] >= 8).astype(int)

        # Combined screening composite
        available_scores = [f for f in self.SCREENING_SCORES if f in df.columns]
        if len(available_scores) > 1:
            # Normalize each score to 0-1 range and average
            normalized_scores = df[available_scores].copy()
            for col in available_scores:
                col_min = normalized_scores[col].min()
                col_max = normalized_scores[col].max()
                if col_max > col_min:
                    normalized_scores[col] = (normalized_scores[col] - col_min) / (col_max - col_min)
            df['combined_screening_score'] = normalized_scores.mean(axis=1)

        return df

    def create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary risk indicator features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with risk indicators
        """
        df = df.copy()
        logger.info("Creating risk indicator features")

        risk_indicators = []

        # High behavioral concern
        if 'behavioral_concern_pct' in df.columns:
            df['high_behavioral_concern'] = (df['behavioral_concern_pct'] > 0.5).astype(int)
            risk_indicators.append('high_behavioral_concern')

        # Communication red flags
        if 'language_regression' in df.columns:
            risk_indicators.append('language_regression')

        # Family history
        if 'family_history_asd' in df.columns:
            risk_indicators.append('family_history_asd')

        # M-CHAT high risk
        if 'mchat_high_risk' in df.columns:
            risk_indicators.append('mchat_high_risk')

        # Total risk indicator count
        available_risk = [r for r in risk_indicators if r in df.columns]
        if available_risk:
            df['total_risk_indicators'] = df[available_risk].sum(axis=1)

        return df

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting comprehensive feature engineering")

        df = self.create_behavioral_composite_scores(df)
        df = self.create_communication_features(df)
        df = self.create_demographic_interactions(df)
        df = self.create_screening_score_features(df)
        df = self.create_risk_indicators(df)

        logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df

    def select_features_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'random_forest',
        n_features: Optional[int] = None,
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Select features based on importance scores.

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method ('random_forest', 'mutual_info', 'f_classif')
            n_features: Number of features to select (None for threshold-based)
            threshold: Minimum importance threshold

        Returns:
            Tuple of (selected features DataFrame, importance dict)
        """
        logger.info(f"Selecting features using {method} method")

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy()

        # Handle any remaining missing values
        X_numeric = X_numeric.fillna(X_numeric.median())

        if method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_numeric, y)
            importances = dict(zip(numeric_cols, rf.feature_importances_))

        elif method == 'mutual_info':
            mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
            importances = dict(zip(numeric_cols, mi_scores))

        elif method == 'f_classif':
            f_scores, _ = f_classif(X_numeric, y)
            # Normalize f_scores
            f_scores = f_scores / (f_scores.max() + 1e-10)
            importances = dict(zip(numeric_cols, f_scores))

        else:
            raise ValueError(f"Unknown method: {method}")

        # Sort by importance
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        self.feature_importances = importances

        # Select features
        if n_features is not None:
            selected = list(importances.keys())[:n_features]
        else:
            selected = [f for f, imp in importances.items() if imp >= threshold]

        self.selected_features = selected
        self._fitted = True

        logger.info(f"Selected {len(selected)} features from {len(numeric_cols)}")

        return X[selected], importances

    def select_features_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 15,
        step: int = 1
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using Recursive Feature Elimination.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            step: Number of features to remove at each step

        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        logger.info(f"Selecting {n_features} features using RFE")

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy().fillna(X[numeric_cols].median())

        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.selector = RFE(estimator, n_features_to_select=n_features, step=step)
        self.selector.fit(X_numeric, y)

        selected_mask = self.selector.support_
        selected_features = [col for col, selected in zip(numeric_cols, selected_mask) if selected]

        self.selected_features = selected_features
        self._fitted = True

        logger.info(f"RFE selected features: {selected_features}")

        return X[selected_features], selected_features

    def remove_low_variance_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Remove features with low variance.

        Args:
            X: Feature DataFrame
            threshold: Variance threshold

        Returns:
            DataFrame with low-variance features removed
        """
        logger.info(f"Removing features with variance < {threshold}")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy().fillna(X[numeric_cols].median())

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_numeric)

        selected_mask = selector.get_support()
        selected_cols = [col for col, selected in zip(numeric_cols, selected_mask) if selected]
        removed_cols = [col for col, selected in zip(numeric_cols, selected_mask) if not selected]

        if removed_cols:
            logger.info(f"Removed low variance features: {removed_cols}")

        # Keep non-numeric columns plus selected numeric columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        return X[non_numeric_cols + selected_cols]

    def apply_pca(
        self,
        X: pd.DataFrame,
        n_components: Union[int, float] = 0.95,
        prefix: str = 'pca'
    ) -> pd.DataFrame:
        """
        Apply PCA dimensionality reduction.

        Args:
            X: Feature DataFrame
            n_components: Number of components or variance ratio to preserve
            prefix: Prefix for PCA column names

        Returns:
            DataFrame with PCA components
        """
        logger.info(f"Applying PCA with n_components={n_components}")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy().fillna(X[numeric_cols].median())

        # Standardize before PCA
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)

        self.pca = PCA(n_components=n_components, random_state=42)
        pca_result = self.pca.fit_transform(X_scaled)

        # Create DataFrame with PCA results
        n_actual_components = pca_result.shape[1]
        pca_cols = [f'{prefix}_{i+1}' for i in range(n_actual_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=X.index)

        logger.info(f"PCA reduced {len(numeric_cols)} features to {n_actual_components} components")
        logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.2%}")

        return pca_df

    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Get a formatted feature importance report.

        Returns:
            DataFrame with feature importance rankings
        """
        if not self.feature_importances:
            logger.warning("No feature importances computed yet")
            return pd.DataFrame()

        report = pd.DataFrame([
            {'feature': feat, 'importance': imp, 'rank': i+1}
            for i, (feat, imp) in enumerate(self.feature_importances.items())
        ])

        report['cumulative_importance'] = report['importance'].cumsum()
        report['importance_pct'] = report['importance'] / report['importance'].sum() * 100

        return report

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save feature engineer state.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'config': self.config,
            'selected_features': self.selected_features,
            'feature_importances': self.feature_importances,
            'selector': self.selector,
            'pca': self.pca,
            'fitted': self._fitted
        }

        joblib.dump(state, filepath)
        logger.info(f"Feature engineer saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FeatureEngineer':
        """
        Load feature engineer state.

        Args:
            filepath: Path to saved file

        Returns:
            Loaded FeatureEngineer instance
        """
        state = joblib.load(filepath)

        engineer = cls(config=state['config'])
        engineer.selected_features = state['selected_features']
        engineer.feature_importances = state['feature_importances']
        engineer.selector = state['selector']
        engineer.pca = state['pca']
        engineer._fitted = state['fitted']

        logger.info(f"Feature engineer loaded from {filepath}")
        return engineer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    engineer = FeatureEngineer()

    # Example with sample data
    # df = pd.read_csv('data/raw/asd_train_data.csv')
    # df_engineered = engineer.engineer_all_features(df)
    # print(f"Engineered features shape: {df_engineered.shape}")
