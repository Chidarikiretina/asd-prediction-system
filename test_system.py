"""
Quick Test Script for ASD Prediction System

Run this script to verify all components are working correctly.
Usage: python test_system.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_data_loading():
    """Test data loading functionality."""
    print("\n" + "="*60)
    print("1. TESTING DATA LOADING")
    print("="*60)

    from data_processing.data_loader import DataLoader

    loader = DataLoader("data/raw")

    # Load training data
    train_df = loader.load_raw_data("asd_train_data.csv")
    print(f"[OK] Loaded training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

    # Load test data
    test_df = loader.load_raw_data("asd_test_data.csv")
    print(f"[OK] Loaded test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    # Validate data
    is_valid, report = loader.validate_data(train_df)
    print(f"[OK] Data validation passed: {is_valid}")

    # Split data
    train_split, test_split = loader.split_data(train_df, test_size=0.2)
    print(f"[OK] Data split: train={len(train_split)}, test={len(test_split)}")

    return train_df, test_df


def test_preprocessing(train_df):
    """Test preprocessing functionality."""
    print("\n" + "="*60)
    print("2. TESTING PREPROCESSING")
    print("="*60)

    from data_processing.preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor()

    # Remove ID column for processing
    df = train_df.drop(columns=['participant_id'], errors='ignore')

    # Run preprocessing pipeline
    X, y = preprocessor.preprocess_pipeline(
        df,
        target_column='asd_diagnosis',
        missing_strategy='median',
        normalize=False,
        handle_outliers=True
    )

    print(f"[OK] Preprocessing complete")
    print(f"     - Features shape: {X.shape}")
    print(f"     - Target shape: {y.shape}")
    print(f"     - Missing values: {X.isnull().sum().sum()}")

    return X, y


def test_feature_engineering(X):
    """Test feature engineering functionality."""
    print("\n" + "="*60)
    print("3. TESTING FEATURE ENGINEERING")
    print("="*60)

    from feature_engineering.feature_engineer import FeatureEngineer

    engineer = FeatureEngineer()

    # Apply feature engineering
    X_engineered = engineer.engineer_all_features(X)

    new_features = X_engineered.shape[1] - X.shape[1]
    print(f"[OK] Feature engineering complete")
    print(f"     - Original features: {X.shape[1]}")
    print(f"     - Engineered features: {X_engineered.shape[1]}")
    print(f"     - New features created: {new_features}")

    return X_engineered


def test_model_training(X, y):
    """Test model training functionality."""
    print("\n" + "="*60)
    print("4. TESTING MODEL TRAINING")
    print("="*60)

    from models.xgboost_model import ASDXGBoostModel
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Prepare numeric features only
    X_numeric = X.select_dtypes(include=[np.number]).fillna(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = ASDXGBoostModel(params={'n_estimators': 50, 'max_depth': 4})
    model.train(X_train, y_train)
    print(f"[OK] Model trained successfully")

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    # Handle case where predict_proba returns 2D array
    if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]  # Take probability of positive class
    print(f"[OK] Predictions made: {len(y_pred)} samples")

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"[OK] Model evaluation:")
    print(f"     - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"     - Precision: {metrics['precision']:.4f}")
    print(f"     - Recall:    {metrics['recall']:.4f}")
    print(f"     - ROC-AUC:   {metrics['roc_auc']:.4f}")

    return model, X_test, y_test, y_pred, y_prob


def test_evaluation(y_test, y_pred, y_prob):
    """Test evaluation functionality."""
    print("\n" + "="*60)
    print("5. TESTING EVALUATION MODULE")
    print("="*60)

    from evaluation.evaluator import ModelEvaluator
    import numpy as np

    evaluator = ModelEvaluator()

    # Calculate metrics
    metrics = evaluator.calculate_metrics(
        y_test.values if hasattr(y_test, 'values') else y_test,
        y_pred,
        y_prob
    )

    print(f"[OK] Comprehensive metrics calculated:")
    print(f"     - Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"     - Specificity: {metrics['specificity']:.4f}")
    print(f"     - PPV:         {metrics['ppv']:.4f}")
    print(f"     - NPV:         {metrics['npv']:.4f}")

    # Check target performance
    targets_met = evaluator.check_target_performance(metrics)
    print(f"\n[OK] Target performance check:")
    for metric, met in targets_met.items():
        status = "MET" if met else "NOT MET"
        print(f"     - {metric}: {status}")

    # Clinical utility
    clinical = evaluator.evaluate_clinical_utility(
        y_test.values if hasattr(y_test, 'values') else y_test,
        y_pred,
        population_size=10000
    )
    print(f"\n[OK] Clinical utility (per 10,000 screened):")
    print(f"     - Cases detected: {clinical['cases_detected']}/{clinical['expected_cases']}")
    print(f"     - False alarms: {clinical['false_alarms']}")

    return evaluator


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("#  ASD PREDICTION SYSTEM - VERIFICATION TEST")
    print("#"*60)

    try:
        # Test each component
        train_df, test_df = test_data_loading()
        X, y = test_preprocessing(train_df)
        X_engineered = test_feature_engineering(X)
        model, X_test, y_test, y_pred, y_prob = test_model_training(X_engineered, y)
        evaluator = test_evaluation(y_test, y_pred, y_prob)

        # Summary
        print("\n" + "="*60)
        print("VERIFICATION COMPLETE - ALL COMPONENTS WORKING!")
        print("="*60)
        print("\nYou can now:")
        print("  1. Run notebooks in notebooks/exploratory/ and notebooks/training/")
        print("  2. Run unit tests: python -m pytest tests/ -v")
        print("  3. Train a full model using the training notebook")
        print("  4. Customize the config in config/config.yaml")

        return True

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
