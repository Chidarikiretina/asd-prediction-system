"""
XGBoost Model Module

This module implements the XGBoost classifier for ASD prediction.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ASDXGBoostModel:
    """
    XGBoost classifier for ASD prediction
    """
    
    def __init__(self, **params):
        """
        Initialize XGBoost model
        
        Args:
            **params: XGBoost parameters
        """
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(params)
        self.model = xgb.XGBClassifier(**default_params)
        self.feature_importance = None
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info("Training XGBoost model")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Model training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0 or 1)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv: int = 5
    ) -> dict:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in scoring:
            scores = cross_val_score(
                self.model, X, y, 
                cv=cv, scoring=metric
            )
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        
        return cv_results
    
    def hyperparameter_tuning(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        param_grid: dict = None,
        cv: int = 5
    ):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            cv: Number of folds for cross-validation
        """
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200, 300],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        
        logger.info("Starting hyperparameter tuning")
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_}")
        
        return grid_search.best_params_, grid_search.best_score_
    
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, top_n: int = None) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            top_n: Return top N features (None for all)
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not available. Train model first.")
            return None
        
        if top_n:
            return self.feature_importance.head(top_n)
        return self.feature_importance


if __name__ == "__main__":
    # Example usage
    model = ASDXGBoostModel()
