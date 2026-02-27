"""
ASD Prediction System - Web Application

A Flask-based web interface for the ASD screening prediction system.
Run with: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models.xgboost_model import ASDXGBoostModel
from data_processing.preprocessor import DataPreprocessor
from feature_engineering.feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'asd-prediction-system-2024'

# Global model
model = None
preprocessor = None
feature_engineer = None
model_loaded = False


# HTML Template embedded directly
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASD Screening Tool - Zimbabwe</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2c5282;
            --secondary-color: #4299e1;
        }
        body { background-color: #f7fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .navbar { background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); }
        .hero-section { background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); color: white; padding: 40px 0; margin-bottom: 30px; }
        .card { border: none; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .card-header { background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); color: white; border-radius: 15px 15px 0 0 !important; padding: 15px 20px; }
        .section-title { color: var(--primary-color); border-bottom: 3px solid var(--secondary-color); padding-bottom: 10px; margin-bottom: 20px; }
        .form-label { font-weight: 600; color: #2d3748; }
        .btn-primary { background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); border: none; padding: 12px 30px; font-size: 1.1rem; border-radius: 25px; }
        .btn-primary:hover { transform: scale(1.05); box-shadow: 0 5px 20px rgba(66, 153, 225, 0.4); }
        .result-card { display: none; margin-top: 30px; }
        .risk-indicator { font-size: 3rem; font-weight: bold; }
        .risk-low { color: #48bb78; }
        .risk-medium { color: #ed8936; }
        .risk-high { color: #f56565; }
        .progress { height: 25px; border-radius: 15px; }
        .question-group { background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px; }
        footer { background: var(--primary-color); color: white; padding: 20px 0; margin-top: 50px; }
        .loading-spinner { display: none; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="bi bi-heart-pulse me-2"></i>ASD Screening Tool</a>
        </div>
    </nav>

    <section class="hero-section">
        <div class="container text-center">
            <h1><i class="bi bi-clipboard2-pulse me-2"></i>Early ASD Screening Tool</h1>
            <p class="lead mb-0">Supporting early detection of Autism Spectrum Disorder in children aged 18-36 months</p>
            <p class="small mt-2">Designed for Healthcare Providers in Zimbabwe</p>
        </div>
    </section>

    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card mb-4">
                    <div class="card-header">
                        <h4 class="mb-0"><i class="bi bi-file-medical me-2"></i>Screening Assessment Form</h4>
                    </div>
                    <div class="card-body">
                        <form id="screeningForm">
                            <h5 class="section-title"><i class="bi bi-person me-2"></i>Child Information</h5>
                            <div class="row mb-4">
                                <div class="col-md-4">
                                    <label class="form-label">Age (months) *</label>
                                    <input type="number" class="form-control" name="age_months" min="18" max="36" required placeholder="18-36" value="24">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Gender *</label>
                                    <select class="form-select" name="gender" required>
                                        <option value="M">Male</option>
                                        <option value="F">Female</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Family History of ASD</label>
                                    <select class="form-select" name="family_history_asd">
                                        <option value="0">No</option>
                                        <option value="1">Yes</option>
                                    </select>
                                </div>
                            </div>

                            <h5 class="section-title"><i class="bi bi-eye me-2"></i>Behavioral Observations</h5>
                            <p class="text-muted small mb-3">Select "Concern" if the child shows difficulty in these areas</p>

                            <div class="row">
                                <div class="col-md-6">
                                    <div class="question-group">
                                        <label class="form-label">Eye Contact</label>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="eye_contact" value="0" checked>
                                            <label class="form-check-label">No Concern</label>
                                        </div>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="eye_contact" value="1">
                                            <label class="form-check-label">Concern</label>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="question-group">
                                        <label class="form-label">Response to Name</label>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="response_to_name" value="0" checked>
                                            <label class="form-check-label">No Concern</label>
                                        </div>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="response_to_name" value="1">
                                            <label class="form-check-label">Concern</label>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="row">
                                <div class="col-md-6">
                                    <div class="question-group">
                                        <label class="form-label">Pointing to Show Interest</label>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="pointing" value="0" checked>
                                            <label class="form-check-label">No Concern</label>
                                        </div>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="pointing" value="1">
                                            <label class="form-check-label">Concern</label>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="question-group">
                                        <label class="form-label">Social Smiling</label>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="social_smile" value="0" checked>
                                            <label class="form-check-label">No Concern</label>
                                        </div>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="social_smile" value="1">
                                            <label class="form-check-label">Concern</label>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="row">
                                <div class="col-md-6">
                                    <div class="question-group">
                                        <label class="form-label">Repetitive Behaviors</label>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="repetitive_behaviors" value="0" checked>
                                            <label class="form-check-label">No Concern</label>
                                        </div>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="repetitive_behaviors" value="1">
                                            <label class="form-check-label">Concern</label>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="question-group">
                                        <label class="form-label">Joint Attention</label>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="joint_attention" value="0" checked>
                                            <label class="form-check-label">No Concern</label>
                                        </div>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="joint_attention" value="1">
                                            <label class="form-check-label">Concern</label>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="row">
                                <div class="col-md-6">
                                    <div class="question-group">
                                        <label class="form-label">Hand Flapping</label>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="hand_flapping" value="0" checked>
                                            <label class="form-check-label">No Concern</label>
                                        </div>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="hand_flapping" value="1">
                                            <label class="form-check-label">Concern</label>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="question-group">
                                        <label class="form-label">Language Regression</label>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="language_regression" value="0" checked>
                                            <label class="form-check-label">No</label>
                                        </div>
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input" type="radio" name="language_regression" value="1">
                                            <label class="form-check-label">Yes</label>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <h5 class="section-title mt-4"><i class="bi bi-chat-dots me-2"></i>Communication</h5>
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label class="form-label">Approximate Word Count</label>
                                    <input type="number" class="form-control" name="word_count" min="0" max="500" value="50" placeholder="Number of words">
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">M-CHAT Score (0-20, optional)</label>
                                    <input type="number" class="form-control" name="mchat_score" min="0" max="20" value="3" placeholder="If available">
                                </div>
                            </div>

                            <div class="text-center mt-4">
                                <button type="submit" class="btn btn-primary btn-lg">
                                    <span class="submit-text"><i class="bi bi-search me-2"></i>Analyze Risk</span>
                                    <span class="loading-spinner spinner-border spinner-border-sm" role="status"></span>
                                </button>
                            </div>
                        </form>
                    </div>
                </div>

                <div class="card result-card" id="resultCard">
                    <div class="card-header">
                        <h4 class="mb-0"><i class="bi bi-clipboard-check me-2"></i>Screening Results</h4>
                    </div>
                    <div class="card-body text-center">
                        <div class="row">
                            <div class="col-md-6">
                                <h5>Risk Level</h5>
                                <div class="risk-indicator" id="riskLevel">-</div>
                            </div>
                            <div class="col-md-6">
                                <h5>Probability Score</h5>
                                <div class="risk-indicator" id="probability">-</div>
                            </div>
                        </div>
                        <div class="my-4">
                            <h6>Risk Probability</h6>
                            <div class="progress">
                                <div class="progress-bar" id="progressBar" role="progressbar" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="alert mt-4" id="recommendation" role="alert">
                            <i class="bi bi-info-circle me-2"></i>
                            <span id="recommendationText">-</span>
                        </div>
                        <div class="alert alert-secondary mt-3">
                            <small><strong>Disclaimer:</strong> This tool is for screening purposes only and does not provide a diagnosis.
                            Please refer to a qualified healthcare professional for comprehensive evaluation.</small>
                        </div>
                        <button class="btn btn-outline-primary mt-3" onclick="resetForm()">
                            <i class="bi bi-arrow-repeat me-2"></i>New Assessment
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="text-center">
        <div class="container">
            <p class="mb-1">ASD Screening Tool for Zimbabwe</p>
            <small>Developed for early detection support in children aged 18-36 months</small>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('screeningForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            document.querySelector('.submit-text').style.display = 'none';
            document.querySelector('.loading-spinner').style.display = 'inline-block';

            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => { data[key] = value; });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                if (result.success) {
                    displayResults(result);
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
            document.querySelector('.submit-text').style.display = 'inline';
            document.querySelector('.loading-spinner').style.display = 'none';
        });

        function displayResults(result) {
            document.getElementById('resultCard').style.display = 'block';
            const riskLevel = document.getElementById('riskLevel');
            riskLevel.textContent = result.risk_level;
            riskLevel.className = 'risk-indicator risk-' + result.risk_level.toLowerCase();
            document.getElementById('probability').textContent = result.probability + '%';
            const progressBar = document.getElementById('progressBar');
            progressBar.style.width = result.probability + '%';
            progressBar.className = 'progress-bar bg-' + result.risk_color;
            const recommendation = document.getElementById('recommendation');
            recommendation.className = 'alert alert-' + result.risk_color + ' mt-4';
            document.getElementById('recommendationText').textContent = result.recommendation;
            document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth' });
        }

        function resetForm() {
            document.getElementById('screeningForm').reset();
            document.getElementById('resultCard').style.display = 'none';
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""


def load_model():
    """Load or train the model."""
    global model, preprocessor, feature_engineer, model_loaded

    try:
        logger.info("Preparing model...")
        data_path = PROJECT_ROOT / 'data' / 'raw' / 'asd_train_data.csv'

        if not data_path.exists():
            logger.error("Training data not found!")
            return

        # Load and prepare data
        df = pd.read_csv(data_path)
        df = df.drop(columns=['participant_id'], errors='ignore')

        # Preprocess
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_pipeline(df, target_column='asd_diagnosis', normalize=False)

        # Feature engineering
        feature_engineer = FeatureEngineer()
        X = feature_engineer.engineer_all_features(X)

        # Select numeric features
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)

        # Train model
        model = ASDXGBoostModel(params={'n_estimators': 100, 'max_depth': 5})
        model.train(X_numeric, y)

        model_loaded = True
        logger.info("Model ready!")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False


@app.route('/')
def home():
    """Home page."""
    return render_template_string(INDEX_HTML)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction."""
    global model, model_loaded

    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    try:
        data = request.get_json()

        # Create DataFrame
        df = pd.DataFrame([data])

        # Convert types
        numeric_fields = ['age_months', 'word_count', 'mchat_score']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)

        binary_fields = ['eye_contact', 'response_to_name', 'pointing', 'social_smile',
                        'repetitive_behaviors', 'joint_attention', 'hand_flapping',
                        'language_regression', 'family_history_asd']
        for field in binary_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0).astype(int)

        # Add missing columns with defaults
        defaults = {
            'social_communication_score': 3, 'rrb_score': 2, 'gestational_weeks': 39,
            'two_word_phrases': 1, 'echolalia': 0, 'pretend_play': 0,
            'unusual_interests': 0, 'toe_walking': 0, 'lines_up_toys': 0, 'upset_by_change': 0
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        # Feature engineering
        df = feature_engineer.engineer_all_features(df)

        # Get numeric features
        X = df.select_dtypes(include=[np.number]).fillna(0)

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)

        if len(probability.shape) > 1:
            prob_asd = float(probability[0, 1])
        else:
            prob_asd = float(probability[0])

        # Determine risk
        if prob_asd >= 0.7:
            risk_level, risk_color = 'High', 'danger'
            recommendation = 'Immediate referral to specialist recommended'
        elif prob_asd >= 0.4:
            risk_level, risk_color = 'Medium', 'warning'
            recommendation = 'Follow-up screening recommended in 1-2 months'
        else:
            risk_level, risk_color = 'Low', 'success'
            recommendation = 'Continue routine developmental monitoring'

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': round(prob_asd * 100, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("=" * 60)
    print("  ASD PREDICTION SYSTEM - WEB APPLICATION")
    print("=" * 60)
    print("\n  Loading model...")

    load_model()

    print("\n  Starting server...")
    print("\n  Open your browser and go to:")
    print("  --> http://localhost:5000")
    print("\n  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
