# ASD Prediction System - Project Structure Documentation

## Overview
This document provides a comprehensive guide to the ASD Prediction System project structure.

## Directory Structure

```
ASD_Prediction_System/
│
├── config/                          # Configuration files
│   └── config.yaml                  # Main configuration file
│
├── data/                            # Data storage (git-ignored)
│   ├── raw/                         # Original, immutable datasets
│   ├── processed/                   # Cleaned and preprocessed data
│   ├── external/                    # External reference datasets
│   └── validation/                  # Validation datasets
│
├── models/                          # Trained models (git-ignored)
│   ├── trained/                     # Production-ready models
│   ├── checkpoints/                 # Training checkpoints
│   └── experimental/                # Experimental model versions
│
├── notebooks/                       # Jupyter notebooks for analysis
│   ├── exploratory/                 # Exploratory data analysis (EDA)
│   ├── preprocessing/               # Data preprocessing notebooks
│   ├── modeling/                    # Model development notebooks
│   └── evaluation/                  # Model evaluation notebooks
│
├── src/                             # Source code
│   ├── __init__.py                  # Package initialization
│   │
│   ├── data_processing/             # Data handling modules
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Data loading and validation
│   │   └── preprocessor.py         # Data preprocessing
│   │
│   ├── feature_engineering/         # Feature creation and selection
│   │   └── __init__.py
│   │
│   ├── models/                      # Model implementations
│   │   ├── __init__.py
│   │   └── xgboost_model.py        # XGBoost classifier
│   │
│   ├── evaluation/                  # Model evaluation utilities
│   │   └── __init__.py
│   │
│   ├── utils/                       # Helper functions
│   │   └── __init__.py
│   │
│   └── api/                         # API endpoints (future)
│       └── __init__.py
│
├── tests/                           # Test suite
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
│
├── docs/                            # Documentation
│   ├── technical/                   # Technical documentation
│   ├── user_guides/                 # User guides
│   └── api_docs/                    # API documentation
│
├── outputs/                         # Generated outputs (git-ignored)
│   ├── reports/                     # Analysis reports
│   ├── visualizations/              # Plots and charts
│   ├── predictions/                 # Prediction results
│   └── logs/                        # Application logs
│
├── scripts/                         # Utility scripts
│   ├── deployment/                  # Deployment scripts
│   ├── training/                    # Model training scripts
│   └── evaluation/                  # Evaluation scripts
│
├── requirements/                    # Python dependencies
│   ├── requirements.txt             # Production dependencies
│   └── requirements-dev.txt         # Development dependencies
│
├── .gitignore                       # Git ignore file
└── README.md                        # Project README

```

## Key Files and Their Purposes

### Configuration
- **config/config.yaml**: Central configuration file containing all project settings, model parameters, and file paths

### Source Code Modules

#### Data Processing
- **src/data_processing/data_loader.py**: Handles loading data from various sources, validation, and train/test splitting
- **src/data_processing/preprocessor.py**: Data cleaning, missing value handling, encoding, normalization

#### Models
- **src/models/xgboost_model.py**: XGBoost classifier implementation with training, prediction, evaluation, and hyperparameter tuning methods

### Requirements
- **requirements/requirements.txt**: Core production dependencies (numpy, pandas, xgboost, etc.)
- **requirements/requirements-dev.txt**: Development tools (jupyter, pytest, black, etc.)

## Workflow

### 1. Data Preparation
```
data/raw/ → src/data_processing/ → data/processed/
```
1. Place raw data in `data/raw/`
2. Use `data_loader.py` to load and validate
3. Use `preprocessor.py` to clean and transform
4. Save processed data to `data/processed/`

### 2. Feature Engineering
```
data/processed/ → src/feature_engineering/ → data/processed/
```
1. Load processed data
2. Create interaction features
3. Perform feature selection
4. Save feature-engineered data

### 3. Model Training
```
data/processed/ → src/models/ → models/trained/
```
1. Load training data
2. Train XGBoost model
3. Perform hyperparameter tuning
4. Save best model to `models/trained/`

### 4. Model Evaluation
```
models/trained/ + data/processed/ → src/evaluation/ → outputs/reports/
```
1. Load trained model and test data
2. Generate predictions
3. Calculate performance metrics
4. Create visualizations
5. Save results to `outputs/`

### 5. Deployment (Future)
```
models/trained/ → src/api/ → Production Environment
```
1. Load production model
2. Create API endpoints
3. Deploy to cloud or local server

## Best Practices

### Version Control
- All code in `src/` should be version controlled
- Data files in `data/` are git-ignored
- Models in `models/` are git-ignored (use model versioning tools like MLflow or DVC)
- `.gitkeep` files maintain directory structure in git

### Code Organization
- Keep modules focused and single-purpose
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Follow PEP 8 style guidelines

### Data Management
- Never modify files in `data/raw/`
- Document all preprocessing steps
- Use configuration files for reproducibility
- Track data versions separately from code

### Model Management
- Save models with version numbers
- Log all hyperparameters used
- Keep track of model performance metrics
- Maintain model cards documenting model details

### Testing
- Write unit tests for all utility functions
- Test data preprocessing pipelines
- Validate model outputs
- Use continuous integration for automated testing

## Getting Started

1. **Setup Environment**
   ```bash
   cd ASD_Prediction_System
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements/requirements-dev.txt
   ```

2. **Add Data**
   ```bash
   # Place your dataset in data/raw/
   cp path/to/dataset.csv data/raw/
   ```

3. **Configure Project**
   ```bash
   # Edit config/config.yaml with your settings
   nano config/config.yaml
   ```

4. **Run Notebooks**
   ```bash
   jupyter lab
   # Navigate to notebooks/ and start exploring
   ```

5. **Train Model**
   ```python
   from src.models.xgboost_model import ASDXGBoostModel
   from src.data_processing.data_loader import DataLoader
   
   # Load and train model
   loader = DataLoader("data/raw")
   # ... (see module docstrings for details)
   ```

## Next Steps

1. Add raw data to `data/raw/`
2. Create exploratory notebook in `notebooks/exploratory/`
3. Implement feature engineering module
4. Train and evaluate baseline model
5. Iterate on model improvements
6. Document findings in `docs/`

## Notes

- This structure is designed to be modular and scalable
- All paths in config.yaml are relative to project root
- Use logging module for all output (not print statements)
- Keep sensitive information in environment variables, not in code

---
*Last updated: January 2026*
