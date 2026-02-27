# ASD Prediction System for Zimbabwe

## Project Overview
Machine learning-based autism spectrum disorder (ASD) prediction system using XGBoost, designed specifically for the Zimbabwean healthcare context.

## Project Structure

```
ASD_Prediction_System/
├── data/                          # Data storage
│   ├── raw/                       # Original, immutable data
│   ├── processed/                 # Cleaned and preprocessed data
│   ├── external/                  # External reference datasets
│   └── validation/                # Validation datasets
│
├── models/                        # Model storage
│   ├── trained/                   # Production-ready models
│   ├── checkpoints/               # Training checkpoints
│   └── experimental/              # Experimental models
│
├── notebooks/                     # Jupyter notebooks
│   ├── exploratory/              # Data exploration notebooks
│   ├── preprocessing/            # Data preprocessing notebooks
│   ├── modeling/                 # Model development notebooks
│   └── evaluation/               # Model evaluation notebooks
│
├── src/                          # Source code
│   ├── data_processing/          # Data loading and preprocessing
│   ├── feature_engineering/      # Feature creation and selection
│   ├── models/                   # Model definitions and training
│   ├── evaluation/               # Model evaluation utilities
│   ├── utils/                    # Helper functions
│   └── api/                      # API endpoints for deployment
│
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── docs/                         # Documentation
│   ├── technical/                # Technical documentation
│   ├── user_guides/              # User guides
│   └── api_docs/                 # API documentation
│
├── config/                       # Configuration files
│
├── outputs/                      # Generated outputs
│   ├── reports/                  # Analysis reports
│   ├── visualizations/           # Plots and charts
│   ├── predictions/              # Prediction results
│   └── logs/                     # Application logs
│
├── scripts/                      # Utility scripts
│   ├── deployment/               # Deployment scripts
│   ├── training/                 # Model training scripts
│   └── evaluation/               # Evaluation scripts
│
└── requirements/                 # Dependencies
    ├── requirements.txt          # Production requirements
    └── requirements-dev.txt      # Development requirements
```

## Getting Started

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ASD_Prediction_System
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements/requirements.txt
```

### Usage

(To be updated as project develops)

## Deploy on Render

1. Push this repository to GitHub.
2. In Render, create a new `Blueprint` and select this repository.
3. Render will detect `render.yaml` and provision:
   - one Python web service
   - one persistent disk mounted at `/var/data`
4. After first deploy, open the service URL and log in.

### Important Notes

- The app uses SQLite. Database file is configured via `ASD_DB_PATH=/var/data/asd_system.db` so data survives redeploys.
- `SECRET_KEY` is generated automatically by Render from `render.yaml`.
- If you want email alerts, set:
  - `EMAIL_ENABLED=true`
  - `SMTP_SERVER`
  - `SMTP_PORT`
  - `SMTP_SENDER_EMAIL`
  - `SMTP_SENDER_PASSWORD`
  - `ADMIN_EMAILS` (comma-separated)

## Project Objectives

1. Develop XGBoost-based ML model for early ASD prediction in children aged 18-36 months
2. Train model using behavioral screening data
3. Identify most predictive behavioral features
4. Evaluate model performance using standard metrics
5. Determine minimum feature set for acceptable accuracy

## Key Features

- Machine learning-based screening tool
- Cultural adaptation for Zimbabwean context
- Offline-capable design for resource-limited settings
- Healthcare provider-friendly interface
- Transparent and explainable predictions (SHAP values)

## Technology Stack

- **Language**: Python 3.9+
- **ML Framework**: XGBoost, scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Explainability**: SHAP
- **API**: FastAPI/Flask (planned)
- **Database**: PostgreSQL/MongoDB (planned)

## Contributing

(Guidelines to be added)

## License

(To be determined)

## Contact

(Contact information to be added)

## Acknowledgments

- Ministry of Health and Child Care, Zimbabwe
- Academic and research institutions
- Healthcare providers and advocacy groups
- Families affected by ASD

---
*Last updated: January 2026*
