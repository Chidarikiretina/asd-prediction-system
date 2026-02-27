# ASD Prediction System - Quick Start Guide

## 🚀 Getting Your Project Running

### Step 1: Environment Setup (5 minutes)

```bash
# Navigate to project
cd ASD_Prediction_System

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements/requirements.txt

# For development (includes Jupyter, testing tools):
pip install -r requirements/requirements-dev.txt
```

### Step 2: Data Preparation

1. **Place your raw data** in `data/raw/`
   - Supported formats: CSV, Excel, JSON
   - Example: `data/raw/asd_screening_data.csv`

2. **Review data requirements**:
   - Behavioral features (eye contact, response to name, etc.)
   - Demographic information (age, gender, location)
   - Target variable (ASD diagnosis: 0/1)

### Step 3: Configuration

Edit `config/config.yaml` to match your data:

```yaml
# Update paths if needed
paths:
  data:
    raw: "data/raw"
    
# Adjust model parameters
model:
  xgboost:
    max_depth: 6
    learning_rate: 0.1
    # ...
```

### Step 4: Exploratory Data Analysis

```bash
# Start Jupyter
jupyter lab

# Create a new notebook in notebooks/exploratory/
# Or use the provided templates
```

**Basic EDA template**:
```python
import pandas as pd
import matplotlib.pyplot as plt
from src.data_processing.data_loader import DataLoader

# Load data
loader = DataLoader("data/raw")
df = loader.load_raw_data("your_data.csv")

# Explore
print(df.info())
print(df.describe())
df.hist(figsize=(15, 10))
plt.show()
```

### Step 5: Data Preprocessing

```python
from src.data_processing.preprocessor import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Handle missing values
df_clean = preprocessor.handle_missing_values(df, strategy='median')

# Encode categorical features
df_encoded = preprocessor.encode_categorical_features(df_clean)

# Save processed data
df_encoded.to_csv('data/processed/processed_data.csv', index=False)
```

### Step 6: Model Training

```python
from src.models.xgboost_model import ASDXGBoostModel
from sklearn.model_selection import train_test_split

# Prepare data
X = df_encoded.drop('target_column', axis=1)
y = df_encoded['target_column']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = ASDXGBoostModel()
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Model Performance: {metrics}")

# Save model
model.save_model('models/trained/asd_xgboost_v1.joblib')
```

### Step 7: Model Evaluation

```python
# Get feature importance
importance = model.get_feature_importance(top_n=10)
print(importance)

# Cross-validation
cv_results = model.cross_validate(X_train, y_train, cv=5)
print(f"Cross-validation ROC-AUC: {cv_results['roc_auc']['mean']:.3f}")

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## 📊 Common Tasks

### Hyperparameter Tuning
```python
# Define parameter grid
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

# Run tuning
best_params, best_score = model.hyperparameter_tuning(
    X_train, y_train, param_grid=param_grid, cv=5
)
```

### Feature Selection
```python
from src.feature_engineering import feature_selector

# Get top N important features
selector = FeatureSelector()
top_features = selector.select_top_features(X_train, y_train, n=15)

# Retrain with selected features only
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]
```

### Saving Results
```python
import pandas as pd
import matplotlib.pyplot as plt

# Save metrics
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('outputs/reports/model_metrics.csv', index=False)

# Save feature importance plot
importance.plot(kind='barh', figsize=(10, 6))
plt.tight_layout()
plt.savefig('outputs/visualizations/feature_importance.png')
```

## 🔧 Troubleshooting

### Issue: Import errors
**Solution**: Make sure you're in the project root directory and have activated the virtual environment

```bash
cd ASD_Prediction_System
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Issue: Missing dependencies
**Solution**: Reinstall requirements
```bash
pip install -r requirements/requirements.txt --upgrade
```

### Issue: Data loading errors
**Solution**: Check file path and format
```python
import os
print(os.path.exists('data/raw/your_file.csv'))  # Should be True
```

### Issue: Model performance is poor
**Checklist**:
- [ ] Is data properly preprocessed?
- [ ] Are there class imbalances? (Use SMOTE or class weights)
- [ ] Have you tried hyperparameter tuning?
- [ ] Are there enough training samples?
- [ ] Is data quality good? (Check for noise, outliers)

## 📚 Next Steps

1. **Experiment with features**: Try creating interaction features
2. **Tune hyperparameters**: Use GridSearchCV for optimization
3. **Compare models**: Try Random Forest, Logistic Regression
4. **Validate culturally**: Test with Zimbabwean data when available
5. **Document findings**: Keep detailed notes in notebooks/

## 🆘 Getting Help

- Check the main README.md for project overview
- Review PROJECT_STRUCTURE.md for detailed structure
- Check module docstrings for API details
- Look at example notebooks in `notebooks/`

## ✅ Quality Checklist

Before finalizing your model:
- [ ] Data is clean and properly preprocessed
- [ ] Model achieves target performance (≥90% sensitivity)
- [ ] Feature importance is documented
- [ ] Cross-validation results are satisfactory
- [ ] Model is saved with version number
- [ ] Performance metrics are logged
- [ ] Visualizations are saved
- [ ] Code is documented
- [ ] Results are reproducible

---
**Happy Modeling! 🎯**
