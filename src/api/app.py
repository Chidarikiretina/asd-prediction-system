"""
ASD Prediction System - Web Application

A Flask-based web interface for the ASD screening prediction system.
Designed for healthcare providers in Zimbabwe to screen children aged 18-36 months.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, make_response, Response, g
from functools import wraps
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import json
import logging
from datetime import datetime, timedelta
import uuid
import io
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import AAA modules
from database.db import get_db_connection, init_database, close_database
from database.migrations import run_migrations, check_migration_status
from auth.authentication import (
    authenticate_user, create_session, validate_session,
    invalidate_session, change_password, cleanup_expired_sessions
)
from auth.authorization import (
    Permission, login_required, permission_required, admin_required,
    get_user_permissions, has_permission, clear_permissions_cache
)
from auth.audit import (
    AuditAction, AuditLogger, audit_action,
    get_audit_logs, get_audit_summary
)
from auth.user_management import UserManager

from models.xgboost_model import ASDXGBoostModel
from data_processing.preprocessor import DataPreprocessor
from feature_engineering.feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__,
            template_folder=str(Path(__file__).parent / 'templates'),
            static_folder=str(Path(__file__).parent / 'static'))

def _env_flag(name: str, default: bool = False) -> bool:
    """Read boolean values from environment variables."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}

# Security configuration
app.secret_key = os.getenv('SECRET_KEY', 'asd-prediction-system-2024-secure-key')
app.config.update(
    SESSION_COOKIE_SECURE=_env_flag('SESSION_COOKIE_SECURE', default=False),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8)
)

# Email configuration (configure these for production)
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'sender_email': os.getenv('SMTP_SENDER_EMAIL', 'asd.screening@health.gov.zw'),
    'sender_password': os.getenv('SMTP_SENDER_PASSWORD', ''),
    'admin_emails': [email.strip() for email in os.getenv('ADMIN_EMAILS', 'admin@health.gov.zw').split(',') if email.strip()],
    'enabled': _env_flag('EMAIL_ENABLED', default=False)
}

# Multi-language translations
TRANSLATIONS = {
    'en': {
        'app_title': 'ASD Early Screening System',
        'welcome': 'Welcome',
        'login': 'Sign In',
        'logout': 'Logout',
        'dashboard': 'Dashboard',
        'new_screening': 'New Screening',
        'history': 'History',
        'about': 'About',
        'high_risk': 'High Risk',
        'medium_risk': 'Medium Risk',
        'low_risk': 'Low Risk',
        'total_screenings': 'Total Screenings',
        'pending_reviews': 'Pending Reviews',
        'child_info': 'Child Information',
        'child_name': 'Child Name',
        'child_id': 'Child ID',
        'age_months': 'Age (Months)',
        'gender': 'Gender',
        'male': 'Male',
        'female': 'Female',
        'behavioral_observations': 'Behavioral Observations',
        'eye_contact': 'Eye Contact',
        'response_to_name': 'Response to Name',
        'pointing': 'Pointing',
        'social_smile': 'Social Smile',
        'normal': 'Normal',
        'concern': 'Concern',
        'submit_screening': 'Submit Screening',
        'screening_result': 'Screening Result',
        'risk_level': 'Risk Level',
        'probability': 'ASD Probability',
        'recommendation': 'Recommendation',
        'export_csv': 'Export CSV',
        'export_excel': 'Export Excel',
        'print_report': 'Print Report',
        'select_language': 'Select Language',
        'high_risk_alert': 'High Risk Case Alert',
        'immediate_referral': 'Immediate referral to specialist recommended.',
        'followup_recommended': 'Follow-up screening recommended in 1-2 months.',
        'continue_monitoring': 'Continue routine developmental monitoring.'
    },
    'sn': {  # Shona
        'app_title': 'Sisitimu Yekuongorora ASD Nekukurumidza',
        'welcome': 'Mauya',
        'login': 'Pinda',
        'logout': 'Buda',
        'dashboard': 'Peji Rekutanga',
        'new_screening': 'Ongorora Mutsva',
        'history': 'Zvakaitwa',
        'about': 'Nezvesisitimu',
        'high_risk': 'Njodzi Yakakura',
        'medium_risk': 'Njodzi Yepakati',
        'low_risk': 'Njodzi Shoma',
        'total_screenings': 'Kuongorora Kwose',
        'pending_reviews': 'Zvinomirira Kuongororwa',
        'child_info': 'Ruzivo Rwemwana',
        'child_name': 'Zita Remwana',
        'child_id': 'Nhamba Yemwana',
        'age_months': 'Makore (Mwedzi)',
        'gender': 'Murume/Mukadzi',
        'male': 'Mukomana',
        'female': 'Musikana',
        'behavioral_observations': 'Kuona Maitiro',
        'eye_contact': 'Kutarisa Mumaziso',
        'response_to_name': 'Kupindura Zita',
        'pointing': 'Kunongedza',
        'social_smile': 'Kuseka Nevanhu',
        'normal': 'Zvakanaka',
        'concern': 'Kunetseka',
        'submit_screening': 'Tumira Kuongorora',
        'screening_result': 'Mhinduro Yekuongorora',
        'risk_level': 'Mwero Wenjodzi',
        'probability': 'Mukana weASD',
        'recommendation': 'Zvinokurudzirwa',
        'export_csv': 'Dhawunirodha CSV',
        'export_excel': 'Dhawunirodha Excel',
        'print_report': 'Prinda Repoti',
        'select_language': 'Sarudza Mutauro',
        'high_risk_alert': 'Yambiro Yenjodzi Yakakura',
        'immediate_referral': 'Kuendeswa kunyanzvi kunokurudzirwa nekukurumidza.',
        'followup_recommended': 'Kuongorora zvakare kunokurudzirwa mumwedzi 1-2.',
        'continue_monitoring': 'Ramba uchitarisa kukura kwemwana.'
    },
    'nd': {  # Ndebele
        'app_title': 'Uhlelo Lokuhlola i-ASD Masinyane',
        'welcome': 'Siyakwamukela',
        'login': 'Ngena',
        'logout': 'Phuma',
        'dashboard': 'Ikhasi Lokuqala',
        'new_screening': 'Ukuhlola Okutsha',
        'history': 'Umlando',
        'about': 'Mayelana',
        'high_risk': 'Ingozi Ephezulu',
        'medium_risk': 'Ingozi Ephakathi',
        'low_risk': 'Ingozi Ephansi',
        'total_screenings': 'Konke Ukuhlola',
        'pending_reviews': 'Okulindele Ukubuyekezwa',
        'child_info': 'Ulwazi Lomntwana',
        'child_name': 'Ibizo Lomntwana',
        'child_id': 'Inombolo Yomntwana',
        'age_months': 'Iminyaka (Izinyanga)',
        'gender': 'Ubulili',
        'male': 'Umfana',
        'female': 'Intombazana',
        'behavioral_observations': 'Ukubona Ukuziphatha',
        'eye_contact': 'Ukubukana Ngamehlo',
        'response_to_name': 'Ukuphendula Ibizo',
        'pointing': 'Ukukhomba',
        'social_smile': 'Ukumoyizela Ebantwini',
        'normal': 'Kukuhle',
        'concern': 'Ukukhathazeka',
        'submit_screening': 'Thumela Ukuhlola',
        'screening_result': 'Umphumela Wokuhlola',
        'risk_level': 'Izinga Lengozi',
        'probability': 'Amathuba e-ASD',
        'recommendation': 'Isincomo',
        'export_csv': 'Landa i-CSV',
        'export_excel': 'Landa i-Excel',
        'print_report': 'Phrinta Umbiko',
        'select_language': 'Khetha Ulimi',
        'high_risk_alert': 'Isexwayiso Sengozi Ephezulu',
        'immediate_referral': 'Ukudluliselwa kongcweti kunconywa masinyane.',
        'followup_recommended': 'Ukuhlola okulandelayo kunconywa ezinyangeni ezi-1-2.',
        'continue_monitoring': 'Qhubeka uqaphe ukukhula komntwana.'
    }
}

def get_translation(key, lang=None):
    """Get translation for a key in the specified language."""
    if lang is None:
        lang = session.get('language', 'en')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

def get_all_translations(lang=None):
    """Get all translations for a language."""
    if lang is None:
        lang = session.get('language', 'en')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en'])

# In-memory screening records storage (in production, use a database)
SCREENING_RECORDS = []


# ============== Database Connection Hooks ==============

@app.before_request
def before_request():
    """Set up database connection before each request."""
    g.db = get_db_connection()
    g.audit_logger = AuditLogger(g.db)


@app.teardown_request
def teardown_request(exception=None):
    """Close database connection after each request."""
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()


# Global model and preprocessor
model = None
preprocessor = None
feature_engineer = None
model_loaded = False
training_columns = None  # Store training feature columns


def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, feature_engineer, model_loaded, training_columns

    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / 'models' / 'trained'

    try:
        # Always train a fresh model to ensure consistency
        logger.info("Training model for web application...")
        train_quick_model()

        model_loaded = True
        logger.info("Model ready for predictions")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False


def train_quick_model():
    """Train a quick model for demonstration."""
    global model, preprocessor, feature_engineer, training_columns

    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'raw' / 'asd_train_data.csv'

    if not data_path.exists():
        logger.error("Training data not found")
        return

    # Load and prepare data
    df = pd.read_csv(data_path)

    # Remove columns that are not collected in the screening form
    # These categorical columns cause one-hot encoding mismatch between training and prediction
    columns_to_remove = ['participant_id', 'geographic_location', 'setting_type',
                         'birth_order', 'ethnicity', 'birth_complications', 'sibling_rank']
    df = df.drop(columns=[c for c in columns_to_remove if c in df.columns], errors='ignore')

    # Convert gender to numeric (F=1, M=0) before preprocessing
    if 'gender' in df.columns:
        gender_map = {'M': 0, 'F': 1, 'male': 0, 'female': 1}
        df['gender'] = df['gender'].map(gender_map).fillna(0).astype(int)

    # Get target column
    y = df['asd_diagnosis'].copy()
    X = df.drop(columns=['asd_diagnosis'])

    # Handle missing values with median for numeric columns
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())

    # Feature engineering
    feature_engineer = FeatureEngineer()
    X = feature_engineer.engineer_all_features(X)

    # Select numeric features
    X_numeric = X.select_dtypes(include=[np.number]).fillna(0)

    # Store training columns for later use
    training_columns = X_numeric.columns.tolist()
    logger.info(f"Training with {len(training_columns)} features: {training_columns[:10]}...")

    # Train model
    model = ASDXGBoostModel(params={'n_estimators': 100, 'max_depth': 5})
    model.train(X_numeric, y)

    logger.info("Quick model trained successfully")


def preprocess_input(data):
    """Preprocess input data for prediction - applying same transformations as training."""
    # Create DataFrame from input
    df = pd.DataFrame([data])

    # Remove non-feature columns that are not used in prediction
    non_feature_cols = ['child_name', 'child_id', 'parent_guardian_name', 'notes',
                        'geographic_location', 'setting_type', 'birth_order',
                        'ethnicity', 'birth_complications', 'sibling_rank']
    df = df.drop(columns=[c for c in non_feature_cols if c in df.columns], errors='ignore')

    # Map M-CHAT-R questions to behavioral features for model compatibility
    # Q1 -> joint_attention (follow gaze)
    # Q2 -> hearing concerns (reverse scored)
    # Q3 -> pretend_play
    # Q4 -> motor activity (climbing)
    # Q5 -> hand_flapping/finger movements (reverse scored)
    # Q6 -> pointing (to ask)
    # Q7 -> pointing (to show interest)
    # Q8 -> social interest
    # Q9 -> social_smile/showing things
    # Q10 -> response_to_name
    # Q11 -> social_smile
    # Q12 -> sensory_sensitivity (sounds - reverse scored)
    # Q13 -> motor development (walking)
    # Q14 -> eye_contact
    # Q15 -> imitation
    # Q16 -> joint_attention
    # Q17 -> attention seeking
    # Q18 -> language comprehension
    # Q19 -> social referencing
    # Q20 -> sensory seeking (movement)

    mchat_mappings = {
        'mchat_q1': 'joint_attention',
        'mchat_q3': 'pretend_play',
        'mchat_q5': 'hand_flapping',
        'mchat_q6': 'pointing',
        'mchat_q7': 'pointing',
        'mchat_q10': 'response_to_name',
        'mchat_q11': 'social_smile',
        'mchat_q12': 'sensory_sensitivity',
        'mchat_q14': 'eye_contact',
        'mchat_q16': 'joint_attention',
    }

    # Apply mappings - take the max concern value for each behavioral feature
    for mchat_q, behavior in mchat_mappings.items():
        if mchat_q in df.columns:
            val = int(df[mchat_q].iloc[0]) if pd.notna(df[mchat_q].iloc[0]) else 0
            if behavior in df.columns:
                df[behavior] = max(int(df[behavior].iloc[0] or 0), val)
            else:
                df[behavior] = val

    # Remove M-CHAT question columns after mapping (they're used for score calculation separately)
    mchat_cols = [f'mchat_q{i}' for i in range(1, 21)]
    df = df.drop(columns=[c for c in mchat_cols if c in df.columns], errors='ignore')

    # Convert to appropriate types
    numeric_fields = ['age_months', 'word_count', 'mchat_score',
                      'social_communication_score', 'rrb_score', 'gestational_weeks']
    binary_fields = ['eye_contact', 'response_to_name', 'pointing', 'social_smile',
                     'repetitive_behaviors', 'joint_attention', 'pretend_play',
                     'unusual_interests', 'hand_flapping', 'toe_walking',
                     'lines_up_toys', 'upset_by_change', 'two_word_phrases',
                     'echolalia', 'language_regression', 'family_history_asd',
                     'sensory_sensitivity', 'motor_delays', 'sleep_issues',
                     'feeding_difficulties', 'stranger_anxiety']

    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)

    for field in binary_fields:
        if field in df.columns:
            # Convert string "0"/"1" or empty to int, default to 0
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0).astype(int)

    # Handle gender encoding (male=0, female=1) - same as training
    if 'gender' in df.columns:
        gender_map = {'male': 0, 'Male': 0, 'M': 0, 'm': 0, '0': 0, 0: 0,
                      'female': 1, 'Female': 1, 'F': 1, 'f': 1, '1': 1, 1: 1}
        df['gender'] = df['gender'].map(gender_map).fillna(0).astype(int)

    return df


def save_screening_record(data, result, user):
    """Save screening record to storage."""
    record = {
        'id': str(uuid.uuid4())[:8].upper(),
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M'),
        'screener': user,
        'screener_username': session.get('user', 'unknown'),
        'facility': session.get('facility', 'Unknown'),
        'child_id': data.get('child_id', f"C{len(SCREENING_RECORDS)+1001}"),
        'child_name': data.get('child_name', 'Anonymous'),
        'age_months': int(data.get('age_months', 0)),
        'gender': data.get('gender', 'Unknown'),
        'risk_level': result['risk_level'],
        'probability': result['probability'],
        'recommendation': result['recommendation'],
        'input_data': data,
        'status': 'Pending Review' if result['risk_level'] in ['High', 'Medium'] else 'Completed'
    }
    SCREENING_RECORDS.insert(0, record)  # Add to beginning

    # Send email notification for high-risk cases
    if result['risk_level'] == 'High':
        send_high_risk_notification(record)

    return record


def get_dashboard_stats():
    """Calculate dashboard statistics."""
    total = len(SCREENING_RECORDS)
    if total == 0:
        return {
            'total_screenings': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0,
            'high_risk_pct': 0,
            'medium_risk_pct': 0,
            'low_risk_pct': 0,
            'today_screenings': 0,
            'pending_reviews': 0,
            'avg_probability': 0
        }

    high = sum(1 for r in SCREENING_RECORDS if r['risk_level'] == 'High')
    medium = sum(1 for r in SCREENING_RECORDS if r['risk_level'] == 'Medium')
    low = sum(1 for r in SCREENING_RECORDS if r['risk_level'] == 'Low')
    today = datetime.now().strftime('%Y-%m-%d')
    today_count = sum(1 for r in SCREENING_RECORDS if r['date'] == today)
    pending = sum(1 for r in SCREENING_RECORDS if r['status'] == 'Pending Review')
    avg_prob = sum(r['probability'] for r in SCREENING_RECORDS) / total

    return {
        'total_screenings': total,
        'high_risk': high,
        'medium_risk': medium,
        'low_risk': low,
        'high_risk_pct': round(high / total * 100, 1) if total > 0 else 0,
        'medium_risk_pct': round(medium / total * 100, 1) if total > 0 else 0,
        'low_risk_pct': round(low / total * 100, 1) if total > 0 else 0,
        'today_screenings': today_count,
        'pending_reviews': pending,
        'avg_probability': round(avg_prob, 1)
    }


# ============== Routes ==============

@app.route('/')
def home():
    """Home page - redirect to login if not authenticated."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))


@app.route('/health')
def health():
    """Simple health check endpoint for platform probes."""
    return jsonify({'status': 'ok'}), 200


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page with AAA authentication."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')

        # Authenticate using new AAA system
        success, user_data, message = authenticate_user(g.db, username, password)

        # Log the attempt
        g.audit_logger.log_login(
            username=username,
            success=success,
            user_id=user_data['id'] if user_data else None,
            error_message=message if not success else None
        )

        if success:
            # Create session
            session_id = create_session(
                g.db,
                user_data['id'],
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string if request.user_agent else None
            )

            # Set session variables
            session.permanent = True
            session['user_id'] = user_data['id']
            session['user'] = user_data['username']
            session['name'] = user_data['name']
            session['role'] = user_data['role_name']
            session['facility'] = user_data['facility']
            session['session_id'] = session_id
            session['must_change_password'] = user_data['must_change_password']

            logger.info(f"User '{username}' logged in successfully")

            # Check if password change is required
            if user_data['must_change_password']:
                flash('You must change your password before continuing.', 'warning')
                return redirect(url_for('change_password_page'))

            flash(f'Welcome, {user_data["name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash(message, 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout and clear session."""
    user_id = session.get('user_id')
    username = session.get('user', 'Unknown')
    session_id = session.get('session_id')

    # Log the logout
    if hasattr(g, 'audit_logger') and user_id:
        g.audit_logger.log_logout(user_id, username)

    # Invalidate database session
    if hasattr(g, 'db') and session_id:
        invalidate_session(g.db, session_id)

    # Clear Flask session
    session.clear()
    logger.info(f"User '{username}' logged out")
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Forgot password page - submit reset request."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        reason = request.form.get('reason', '').strip()

        if not username:
            flash('Please enter your username.', 'danger')
            return render_template('forgot_password.html')

        # Check if user exists
        cursor = g.db.cursor()
        cursor.execute('SELECT id, username, name FROM users WHERE username = ? AND is_active = 1', (username,))
        user = cursor.fetchone()

        if not user:
            # Don't reveal if user exists or not for security
            flash('If an account with that username exists, a password reset request has been submitted.', 'info')
            return render_template('forgot_password.html')

        # Check for existing pending request
        cursor.execute('''
            SELECT id FROM password_reset_requests
            WHERE user_id = ? AND status = 'pending'
            ORDER BY created_at DESC LIMIT 1
        ''', (user['id'],))
        existing = cursor.fetchone()

        if existing:
            flash('A password reset request is already pending for this account. Please wait for an administrator to process it.', 'warning')
            return render_template('forgot_password.html')

        # Create reset request
        cursor.execute('''
            INSERT INTO password_reset_requests (user_id, username, reason, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        ''', (user['id'], username, reason, request.remote_addr,
              request.user_agent.string if request.user_agent else None))
        g.db.commit()

        # Log the request
        g.audit_logger.log(
            action_type='PASSWORD_RESET_REQUEST',
            resource_type='auth',
            resource_id=str(user['id']),
            success=True,
            details={'username': username}
        )

        flash('Password reset request submitted successfully. An administrator will review your request.', 'success')
        return redirect(url_for('login'))

    return render_template('forgot_password.html')


@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password_page():
    """Password change page."""
    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
            return render_template('change_password.html', user=session.get('name', 'User'))

        success, message = change_password(
            g.db,
            session.get('user_id'),
            current_password,
            new_password
        )

        if success:
            # Log the password change
            g.audit_logger.log(
                action_type=AuditAction.PASSWORD_CHANGE,
                resource_type='auth',
                success=True
            )
            session['must_change_password'] = False
            flash('Password changed successfully.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash(message, 'danger')

    return render_template('change_password.html', user=session.get('name', 'User'))


@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard with statistics."""
    stats = get_dashboard_stats()
    recent_records = SCREENING_RECORDS[:5]  # Last 5 records
    return render_template('dashboard.html',
                         user=session.get('name', 'User'),
                         stats=stats,
                         recent_records=recent_records)


@app.route('/screening')
@login_required
@permission_required(Permission.SCREENING_CREATE)
def screening():
    """Screening form page - requires login and screening.create permission."""
    return render_template('index.html', user=session.get('name', 'User'))


@app.route('/history')
@login_required
@permission_required(Permission.SCREENING_VIEW)
def history():
    """Screening history page."""
    # Check if user can view all screenings
    can_view_all = has_permission(g.db, session.get('user_id'), Permission.SCREENING_VIEW_ALL)

    if can_view_all:
        records = SCREENING_RECORDS
    else:
        records = [r for r in SCREENING_RECORDS if r['screener_username'] == session.get('user')]

    return render_template('history.html',
                         user=session.get('name', 'User'),
                         records=records)


@app.route('/record/<record_id>')
@login_required
@permission_required(Permission.SCREENING_VIEW)
@audit_action(AuditAction.SCREENING_VIEW, 'screening', get_resource_id=lambda *args, **kwargs: kwargs.get('record_id'))
def view_record(record_id):
    """View single screening record."""
    record = next((r for r in SCREENING_RECORDS if r['id'] == record_id), None)
    if not record:
        flash('Record not found.', 'danger')
        return redirect(url_for('history'))

    # Check access rights
    can_view_all = has_permission(g.db, session.get('user_id'), Permission.SCREENING_VIEW_ALL)
    if not can_view_all and record['screener_username'] != session.get('user'):
        flash('You do not have permission to view this record.', 'danger')
        return redirect(url_for('history'))

    return render_template('record_detail.html',
                         user=session.get('name', 'User'),
                         record=record)


@app.route('/predict', methods=['POST'])
@login_required
@permission_required(Permission.SCREENING_CREATE)
def predict():
    """Handle prediction request."""
    global model, model_loaded, training_columns

    if not model_loaded:
        load_model()

    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not available. Please try again later.'
        })

    try:
        # Get form data
        data = request.get_json() if request.is_json else request.form.to_dict()
        logger.info(f"Received input data: {data}")

        # Preprocess input (applies same transformations as training)
        df = preprocess_input(data)
        logger.info(f"After preprocess_input columns: {df.columns.tolist()}")

        # Apply feature engineering
        if feature_engineer:
            df = feature_engineer.engineer_all_features(df)
            logger.info(f"After feature engineering columns: {df.columns.tolist()}")

        # Select numeric features and fill missing
        X = df.select_dtypes(include=[np.number]).fillna(0)
        logger.info(f"Numeric features before alignment: {X.columns.tolist()}")

        # Align columns with training data
        if training_columns:
            # Add missing columns with default value 0
            for col in training_columns:
                if col not in X.columns:
                    X[col] = 0
            # Keep only training columns in the correct order
            X = X[training_columns]
            logger.info(f"Features after alignment: {X.shape}")

        # Log feature values for debugging
        non_zero_features = X.iloc[0][X.iloc[0] != 0]
        logger.info(f"Input features summary - non-zero cols ({len(non_zero_features)}): {non_zero_features.to_dict()}")

        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)

        # Handle 2D probability array
        if len(probability.shape) > 1:
            prob_asd = float(probability[0, 1])
        else:
            prob_asd = float(probability[0])

        # Determine risk level
        if prob_asd >= 0.7:
            risk_level = 'High'
            risk_color = 'danger'
            recommendation = 'Immediate referral to specialist recommended. Please arrange comprehensive developmental evaluation.'
        elif prob_asd >= 0.4:
            risk_level = 'Medium'
            risk_color = 'warning'
            recommendation = 'Follow-up screening recommended in 1-2 months. Monitor developmental milestones closely.'
        else:
            risk_level = 'Low'
            risk_color = 'success'
            recommendation = 'Continue routine developmental monitoring. No immediate concerns identified.'

        result = {
            'success': True,
            'prediction': int(prediction),
            'probability': round(prob_asd * 100, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation
        }

        # Save screening record
        record = save_screening_record(data, result, session.get('name', 'User'))
        result['record_id'] = record['id']

        # Log the screening creation
        g.audit_logger.log_screening(
            action=AuditAction.SCREENING_CREATE,
            screening_id=record['id'],
            details={'risk_level': risk_level, 'probability': round(prob_asd * 100, 1)}
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/report/<record_id>')
@login_required
def generate_report(record_id):
    """Generate PDF-style report (HTML for now)."""
    record = next((r for r in SCREENING_RECORDS if r['id'] == record_id), None)
    if not record:
        return jsonify({'error': 'Record not found'}), 404

    return render_template('report.html', record=record)


@app.route('/api/stats')
@login_required
def api_stats():
    """API endpoint for dashboard stats."""
    return jsonify(get_dashboard_stats())


@app.route('/api/analytics')
@login_required
def api_analytics():
    """API endpoint for detailed analytics data (age, gender, monthly)."""
    from collections import defaultdict

    # Age distribution
    age_groups = {'18-22m': 0, '23-27m': 0, '28-32m': 0, '33-36m': 0}
    for record in SCREENING_RECORDS:
        age = record.get('age_months', 0)
        if age <= 22:
            age_groups['18-22m'] += 1
        elif age <= 27:
            age_groups['23-27m'] += 1
        elif age <= 32:
            age_groups['28-32m'] += 1
        else:
            age_groups['33-36m'] += 1

    # Gender distribution
    gender_counts = defaultdict(int)
    for record in SCREENING_RECORDS:
        g_val = record.get('gender', 'Unknown')
        if g_val in ('Male', 'male', 'M'):
            gender_counts['Male'] += 1
        elif g_val in ('Female', 'female', 'F'):
            gender_counts['Female'] += 1
        else:
            gender_counts['Unknown'] += 1

    # Monthly trend (last 6 months)
    monthly = defaultdict(int)
    for record in SCREENING_RECORDS:
        month_key = record.get('date', '')[:7]  # YYYY-MM
        if month_key:
            monthly[month_key] += 1
    sorted_months = sorted(monthly.keys())[-6:]

    return jsonify({
        'age_groups': {
            'labels': list(age_groups.keys()),
            'counts': list(age_groups.values())
        },
        'gender': {
            'labels': list(gender_counts.keys()) if gender_counts else ['No data'],
            'counts': list(gender_counts.values()) if gender_counts else [0]
        },
        'monthly': {
            'labels': sorted_months if sorted_months else ['No data'],
            'counts': [monthly[m] for m in sorted_months] if sorted_months else [0]
        }
    })


@app.route('/api/extend-session')
@login_required
def api_extend_session():
    """Extend the current user session."""
    session.modified = True
    return jsonify({'status': 'ok', 'message': 'Session extended'})


@app.route('/api/chart-data')
@login_required
def api_chart_data():
    """API endpoint for chart data."""
    # Get last 7 days data
    from collections import defaultdict
    daily_counts = defaultdict(lambda: {'high': 0, 'medium': 0, 'low': 0})

    for record in SCREENING_RECORDS:
        date = record['date']
        risk = record['risk_level'].lower()
        daily_counts[date][risk] += 1

    # Sort by date
    sorted_dates = sorted(daily_counts.keys())[-7:]

    return jsonify({
        'labels': sorted_dates,
        'high': [daily_counts[d]['high'] for d in sorted_dates],
        'medium': [daily_counts[d]['medium'] for d in sorted_dates],
        'low': [daily_counts[d]['low'] for d in sorted_dates]
    })


@app.route('/patients')
@login_required
@permission_required(Permission.SCREENING_VIEW)
def patients():
    """Patient management - list all children screened."""
    can_view_all = has_permission(g.db, session.get('user_id'), Permission.SCREENING_VIEW_ALL)

    if can_view_all:
        records = SCREENING_RECORDS
    else:
        records = [r for r in SCREENING_RECORDS if r['screener_username'] == session.get('user')]

    # Build unique patient list with latest screening info
    patient_map = {}
    for record in records:
        cid = record.get('child_id', '')
        if cid not in patient_map:
            patient_map[cid] = {
                'child_id': cid,
                'child_name': record.get('child_name', 'Unknown'),
                'age_months': record.get('age_months', 0),
                'gender': record.get('gender', 'Unknown'),
                'total_screenings': 0,
                'latest_risk': record.get('risk_level', 'Unknown'),
                'latest_probability': record.get('probability', 0),
                'latest_date': record.get('date', ''),
                'latest_record_id': record.get('id', ''),
                'status': record.get('status', 'Unknown'),
                'father_name': record.get('input_data', {}).get('father_name', ''),
                'mother_name': record.get('input_data', {}).get('mother_name', ''),
                'screenings': []
            }
        patient_map[cid]['total_screenings'] += 1
        patient_map[cid]['screenings'].append({
            'id': record.get('id'),
            'date': record.get('date'),
            'risk_level': record.get('risk_level'),
            'probability': record.get('probability')
        })

    patients_list = sorted(patient_map.values(), key=lambda x: x['latest_date'], reverse=True)

    return render_template('patients.html',
                         user=session.get('name', 'User'),
                         patients=patients_list)


@app.route('/patients/<child_id>')
@login_required
@permission_required(Permission.SCREENING_VIEW)
def patient_detail(child_id):
    """View patient profile with all screening history."""
    can_view_all = has_permission(g.db, session.get('user_id'), Permission.SCREENING_VIEW_ALL)

    if can_view_all:
        records = [r for r in SCREENING_RECORDS if r.get('child_id') == child_id]
    else:
        records = [r for r in SCREENING_RECORDS if r.get('child_id') == child_id and r['screener_username'] == session.get('user')]

    if not records:
        flash('Patient not found.', 'danger')
        return redirect(url_for('patients'))

    latest = records[0]
    patient = {
        'child_id': child_id,
        'child_name': latest.get('child_name', 'Unknown'),
        'age_months': latest.get('age_months', 0),
        'gender': latest.get('gender', 'Unknown'),
        'father_name': latest.get('input_data', {}).get('father_name', ''),
        'mother_name': latest.get('input_data', {}).get('mother_name', ''),
        'total_screenings': len(records),
        'screenings': records
    }

    return render_template('patient_detail.html',
                         user=session.get('name', 'User'),
                         patient=patient)


@app.route('/patients/<child_id>/add-followup', methods=['POST'])
@login_required
@permission_required(Permission.SCREENING_CREATE)
def add_followup(child_id):
    """Add a follow-up note for a patient."""
    note = request.form.get('note', '').strip()
    followup_date = request.form.get('followup_date', '')

    if not note:
        flash('Please enter a follow-up note.', 'danger')
        return redirect(url_for('patient_detail', child_id=child_id))

    # Find the patient's latest record and add follow-up info
    for record in SCREENING_RECORDS:
        if record.get('child_id') == child_id:
            if 'followups' not in record:
                record['followups'] = []
            record['followups'].append({
                'note': note,
                'date': followup_date or datetime.now().strftime('%Y-%m-%d'),
                'added_by': session.get('name', 'User'),
                'added_at': datetime.now().isoformat()
            })
            break

    g.audit_logger.log(
        action_type='FOLLOWUP_ADDED',
        resource_type='patient',
        resource_id=child_id,
        success=True,
        details={'note': note[:100]}
    )

    flash('Follow-up note added successfully.', 'success')
    return redirect(url_for('patient_detail', child_id=child_id))


@app.route('/api/report/<record_id>/pdf')
@login_required
def generate_pdf_report(record_id):
    """Generate a printable PDF-style HTML report for a screening record."""
    record = next((r for r in SCREENING_RECORDS if r['id'] == record_id), None)
    if not record:
        return jsonify({'error': 'Record not found'}), 404

    g.audit_logger.log(
        action_type='REPORT_GENERATED',
        resource_type='screening',
        resource_id=record_id,
        success=True
    )

    return render_template('report_pdf.html', record=record)


@app.route('/about')
@login_required
def about():
    """About page."""
    return render_template('about.html', user=session.get('name', 'User'))


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })


# ============== Data Export Routes ==============

@app.route('/api/export/csv')
@login_required
@permission_required(Permission.SCREENING_EXPORT)
@audit_action(AuditAction.DATA_EXPORT_CSV, 'screening')
def export_csv():
    """Export screening records to CSV."""
    # Check if user can view all screenings
    can_view_all = has_permission(g.db, session.get('user_id'), Permission.SCREENING_VIEW_ALL)

    if can_view_all:
        records = SCREENING_RECORDS
    else:
        records = [r for r in SCREENING_RECORDS if r['screener_username'] == session.get('user')]

    if not records:
        flash('No records to export.', 'warning')
        return redirect(url_for('history'))

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        'Record ID', 'Date', 'Time', 'Child Name', 'Child ID', 'Age (Months)',
        'Gender', 'Father Name', 'Mother Name', 'Risk Level', 'Probability (%)',
        'M-CHAT Score', 'Recommendation', 'Screener', 'Facility', 'Status'
    ])

    # Data rows
    for record in records:
        writer.writerow([
            record['id'],
            record['date'],
            record['time'],
            record['child_name'],
            record['child_id'],
            record['age_months'],
            record['gender'],
            record.get('input_data', {}).get('father_name', ''),
            record.get('input_data', {}).get('mother_name', ''),
            record['risk_level'],
            record['probability'],
            record.get('input_data', {}).get('mchat_score', ''),
            record['recommendation'],
            record['screener'],
            record['facility'],
            record['status']
        ])

    # Create response
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=asd_screening_records_{datetime.now().strftime("%Y%m%d")}.csv'
        }
    )


@app.route('/api/export/excel')
@login_required
@permission_required(Permission.SCREENING_EXPORT)
@audit_action(AuditAction.DATA_EXPORT_EXCEL, 'screening')
def export_excel():
    """Export screening records to Excel."""
    # Check if user can view all screenings
    can_view_all = has_permission(g.db, session.get('user_id'), Permission.SCREENING_VIEW_ALL)

    if can_view_all:
        records = SCREENING_RECORDS
    else:
        records = [r for r in SCREENING_RECORDS if r['screener_username'] == session.get('user')]

    if not records:
        flash('No records to export.', 'warning')
        return redirect(url_for('history'))

    # Create DataFrame
    df_data = []
    for record in records:
        df_data.append({
            'Record ID': record['id'],
            'Date': record['date'],
            'Time': record['time'],
            'Child Name': record['child_name'],
            'Child ID': record['child_id'],
            'Age (Months)': record['age_months'],
            'Gender': record['gender'],
            'Father Name': record.get('input_data', {}).get('father_name', ''),
            'Mother Name': record.get('input_data', {}).get('mother_name', ''),
            'Risk Level': record['risk_level'],
            'Probability (%)': record['probability'],
            'M-CHAT Score': record.get('input_data', {}).get('mchat_score', ''),
            'Recommendation': record['recommendation'],
            'Screener': record['screener'],
            'Facility': record['facility'],
            'Status': record['status']
        })

    df = pd.DataFrame(df_data)

    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Screening Records', index=False)

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={
            'Content-Disposition': f'attachment; filename=asd_screening_records_{datetime.now().strftime("%Y%m%d")}.xlsx'
        }
    )


# ============== Email Notification Functions ==============

def send_high_risk_notification(record):
    """Send email notification for high-risk cases."""
    if not EMAIL_CONFIG['enabled']:
        logger.info("Email notifications disabled - skipping high-risk alert")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[URGENT] High Risk ASD Screening Alert - {record['child_id']}"
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = ', '.join(EMAIL_CONFIG['admin_emails'])

        # Plain text version
        text_content = f"""
HIGH RISK ASD SCREENING ALERT

A high-risk case has been identified and requires immediate attention.

SCREENING DETAILS:
- Record ID: {record['id']}
- Date: {record['date']} {record['time']}
- Facility: {record['facility']}
- Screener: {record['screener']}

CHILD INFORMATION:
- Name: {record['child_name']}
- ID: {record['child_id']}
- Age: {record['age_months']} months
- Gender: {record['gender']}

ASSESSMENT RESULT:
- Risk Level: {record['risk_level']}
- ASD Probability: {record['probability']}%
- Recommendation: {record['recommendation']}

Please ensure this case receives appropriate follow-up care.

---
ASD Early Screening System
Ministry of Health - Zimbabwe
        """

        # HTML version
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <div style="background: #ef4444; color: white; padding: 20px; text-align: center;">
                <h1 style="margin: 0;">HIGH RISK ASD SCREENING ALERT</h1>
            </div>
            <div style="padding: 20px;">
                <p><strong>A high-risk case has been identified and requires immediate attention.</strong></p>

                <h3 style="color: #1e3a5f;">Screening Details</h3>
                <table style="border-collapse: collapse; width: 100%;">
                    <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Record ID:</strong></td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{record['id']}</td></tr>
                    <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Date/Time:</strong></td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{record['date']} {record['time']}</td></tr>
                    <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Facility:</strong></td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{record['facility']}</td></tr>
                    <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Screener:</strong></td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{record['screener']}</td></tr>
                </table>

                <h3 style="color: #1e3a5f;">Child Information</h3>
                <table style="border-collapse: collapse; width: 100%;">
                    <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Name:</strong></td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{record['child_name']}</td></tr>
                    <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>ID:</strong></td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{record['child_id']}</td></tr>
                    <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Age:</strong></td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{record['age_months']} months</td></tr>
                    <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Gender:</strong></td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{record['gender']}</td></tr>
                </table>

                <h3 style="color: #1e3a5f;">Assessment Result</h3>
                <div style="background: #fef2f2; border: 1px solid #ef4444; border-radius: 8px; padding: 15px;">
                    <p><strong>Risk Level:</strong> <span style="color: #ef4444; font-weight: bold;">{record['risk_level']}</span></p>
                    <p><strong>ASD Probability:</strong> {record['probability']}%</p>
                    <p><strong>Recommendation:</strong> {record['recommendation']}</p>
                </div>

                <p style="margin-top: 20px;"><em>Please ensure this case receives appropriate follow-up care.</em></p>
            </div>
            <div style="background: #1e3a5f; color: white; padding: 15px; text-align: center; font-size: 12px;">
                ASD Early Screening System | Ministry of Health - Zimbabwe
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))

        # Send email
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)

        logger.info(f"High-risk notification sent for record {record['id']}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        return False


# ============== Language Routes ==============

@app.route('/set-language/<lang>')
def set_language(lang):
    """Set the user's preferred language."""
    if lang in TRANSLATIONS:
        session['language'] = lang
        flash(f'Language changed to {lang.upper()}', 'success')
    return redirect(request.referrer or url_for('dashboard'))


@app.route('/api/translations')
def get_translations_api():
    """API endpoint to get translations for current language."""
    lang = session.get('language', 'en')
    return jsonify(TRANSLATIONS.get(lang, TRANSLATIONS['en']))


# Make translations and permissions available to all templates
@app.context_processor
def inject_context():
    """Inject translations and permissions into all templates."""
    lang = session.get('language', 'en')

    # Get user permissions if logged in
    user_permissions = []
    is_admin = False
    can_export = False

    if 'user_id' in session and hasattr(g, 'db'):
        try:
            user_permissions = get_user_permissions(g.db, session.get('user_id'))
            is_admin = session.get('role') == 'admin'
            can_export = Permission.SCREENING_EXPORT in user_permissions
        except Exception:
            pass

    return {
        't': TRANSLATIONS.get(lang, TRANSLATIONS['en']),
        'current_lang': lang,
        'now_date': datetime.now().strftime('%Y-%m-%d'),
        'session_timeout': 8,  # hours
        'available_languages': [
            {'code': 'en', 'name': 'English'},
            {'code': 'sn', 'name': 'Shona'},
            {'code': 'nd', 'name': 'Ndebele'}
        ],
        'is_admin': is_admin,
        'can_export': can_export,
        'user_permissions': user_permissions
    }


# ============== Admin Routes ==============

@app.route('/admin/users')
@login_required
@admin_required
@audit_action(AuditAction.USER_VIEW, 'user')
def admin_users():
    """Admin user management page."""
    user_manager = UserManager(g.db)
    users = user_manager.list_users(include_inactive=True)
    roles = user_manager.get_roles()
    stats = user_manager.get_user_statistics()

    return render_template('admin/users.html',
                         user=session.get('name', 'User'),
                         users=users,
                         roles=roles,
                         stats=stats)


@app.route('/admin/users/create', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_create_user():
    """Create new user."""
    user_manager = UserManager(g.db)

    if request.method == 'POST':
        success, message, user_id = user_manager.create_user(
            username=request.form.get('username', '').strip().lower(),
            password=request.form.get('password', ''),
            name=request.form.get('name', ''),
            role_name=request.form.get('role', ''),
            facility=request.form.get('facility', ''),
            email=request.form.get('email', ''),
            created_by=session.get('user_id'),
            must_change_password=True
        )

        if success:
            g.audit_logger.log_user_management(
                action=AuditAction.USER_CREATE,
                target_user_id=user_id,
                details={'username': request.form.get('username')}
            )
            flash(message, 'success')
            return redirect(url_for('admin_users'))
        else:
            flash(message, 'danger')

    roles = user_manager.get_roles()
    return render_template('admin/user_form.html',
                         user=session.get('name', 'User'),
                         roles=roles,
                         edit_user=None,
                         action='Create')


@app.route('/admin/users/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_user(user_id):
    """Edit user details."""
    user_manager = UserManager(g.db)
    edit_user = user_manager.get_user(user_id)

    if not edit_user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_users'))

    if request.method == 'POST':
        success, message = user_manager.update_user(
            user_id=user_id,
            name=request.form.get('name'),
            email=request.form.get('email'),
            role_name=request.form.get('role'),
            facility=request.form.get('facility')
        )

        if success:
            g.audit_logger.log_user_management(
                action=AuditAction.USER_UPDATE,
                target_user_id=user_id,
                details={'fields_updated': ['name', 'email', 'role', 'facility']}
            )
            flash(message, 'success')
            return redirect(url_for('admin_users'))
        else:
            flash(message, 'danger')

    roles = user_manager.get_roles()
    return render_template('admin/user_form.html',
                         user=session.get('name', 'User'),
                         roles=roles,
                         edit_user=edit_user,
                         action='Edit')


@app.route('/admin/users/<int:user_id>/reset-password', methods=['POST'])
@login_required
@admin_required
def admin_reset_password(user_id):
    """Reset user password."""
    user_manager = UserManager(g.db)
    success, message, temp_password = user_manager.reset_password(user_id)

    if success:
        g.audit_logger.log_user_management(
            action=AuditAction.PASSWORD_RESET,
            target_user_id=user_id
        )
        if temp_password:
            flash(f'{message} Temporary password: {temp_password}', 'success')
        else:
            flash(message, 'success')
    else:
        flash(message, 'danger')

    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/toggle-status', methods=['POST'])
@login_required
@admin_required
def admin_toggle_user_status(user_id):
    """Activate or deactivate user."""
    user_manager = UserManager(g.db)
    edit_user = user_manager.get_user(user_id)

    if not edit_user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_users'))

    if edit_user['is_active']:
        success, message = user_manager.deactivate_user(user_id)
        action = AuditAction.USER_DEACTIVATE
    else:
        success, message = user_manager.reactivate_user(user_id)
        action = AuditAction.USER_REACTIVATE

    if success:
        g.audit_logger.log_user_management(action=action, target_user_id=user_id)
        flash(message, 'success')
    else:
        flash(message, 'danger')

    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/unlock', methods=['POST'])
@login_required
@admin_required
def admin_unlock_user(user_id):
    """Unlock a locked user account."""
    user_manager = UserManager(g.db)
    success, message = user_manager.unlock_account(user_id)

    if success:
        g.audit_logger.log_user_management(
            action=AuditAction.ACCOUNT_UNLOCKED,
            target_user_id=user_id
        )
        flash(message, 'success')
    else:
        flash(message, 'danger')

    return redirect(url_for('admin_users'))


@app.route('/admin/password-requests')
@login_required
@admin_required
def admin_password_requests():
    """View and manage password reset requests."""
    cursor = g.db.cursor()

    # Get pending requests
    cursor.execute('''
        SELECT pr.*, u.name as user_name, u.email as user_email, u.facility
        FROM password_reset_requests pr
        JOIN users u ON pr.user_id = u.id
        WHERE pr.status = 'pending'
        ORDER BY pr.created_at DESC
    ''')
    pending_requests = [dict(row) for row in cursor.fetchall()]

    # Get recent processed requests (last 7 days)
    cursor.execute('''
        SELECT pr.*, u.name as user_name, u.email as user_email,
               admin.name as processed_by_name
        FROM password_reset_requests pr
        JOIN users u ON pr.user_id = u.id
        LEFT JOIN users admin ON pr.processed_by = admin.id
        WHERE pr.status != 'pending'
        AND pr.processed_at > datetime('now', '-7 days')
        ORDER BY pr.processed_at DESC
    ''')
    processed_requests = [dict(row) for row in cursor.fetchall()]

    return render_template('admin/password_requests.html',
                         user=session.get('name', 'User'),
                         pending_requests=pending_requests,
                         processed_requests=processed_requests)


@app.route('/admin/password-requests/<int:request_id>/approve', methods=['POST'])
@login_required
@admin_required
def admin_approve_reset(request_id):
    """Approve a password reset request."""
    cursor = g.db.cursor()

    # Get the request
    cursor.execute('SELECT * FROM password_reset_requests WHERE id = ?', (request_id,))
    reset_request = cursor.fetchone()

    if not reset_request:
        flash('Reset request not found.', 'danger')
        return redirect(url_for('admin_password_requests'))

    if reset_request['status'] != 'pending':
        flash('This request has already been processed.', 'warning')
        return redirect(url_for('admin_password_requests'))

    # Reset the password using user manager
    user_manager = UserManager(g.db)
    success, message, temp_password = user_manager.reset_password(reset_request['user_id'])

    if success:
        # Update request status
        admin_notes = request.form.get('admin_notes', '')
        cursor.execute('''
            UPDATE password_reset_requests
            SET status = 'approved', processed_at = CURRENT_TIMESTAMP,
                processed_by = ?, admin_notes = ?
            WHERE id = ?
        ''', (session.get('user_id'), admin_notes, request_id))
        g.db.commit()

        # Log the action
        g.audit_logger.log_user_management(
            action=AuditAction.PASSWORD_RESET,
            target_user_id=reset_request['user_id'],
            details={'request_id': request_id}
        )

        flash(f'Password reset approved. Temporary password: {temp_password}', 'success')
    else:
        flash(f'Failed to reset password: {message}', 'danger')

    return redirect(url_for('admin_password_requests'))


@app.route('/admin/password-requests/<int:request_id>/reject', methods=['POST'])
@login_required
@admin_required
def admin_reject_reset(request_id):
    """Reject a password reset request."""
    cursor = g.db.cursor()

    # Get the request
    cursor.execute('SELECT * FROM password_reset_requests WHERE id = ?', (request_id,))
    reset_request = cursor.fetchone()

    if not reset_request:
        flash('Reset request not found.', 'danger')
        return redirect(url_for('admin_password_requests'))

    if reset_request['status'] != 'pending':
        flash('This request has already been processed.', 'warning')
        return redirect(url_for('admin_password_requests'))

    # Update request status
    admin_notes = request.form.get('admin_notes', '')
    cursor.execute('''
        UPDATE password_reset_requests
        SET status = 'rejected', processed_at = CURRENT_TIMESTAMP,
            processed_by = ?, admin_notes = ?
        WHERE id = ?
    ''', (session.get('user_id'), admin_notes, request_id))
    g.db.commit()

    # Log the action
    g.audit_logger.log(
        action_type='PASSWORD_RESET_REJECTED',
        resource_type='auth',
        resource_id=str(reset_request['user_id']),
        success=True,
        details={'request_id': request_id, 'reason': admin_notes}
    )

    flash('Password reset request rejected.', 'info')
    return redirect(url_for('admin_password_requests'))


@app.route('/admin/activity')
@login_required
@admin_required
def admin_activity():
    """View active user sessions."""
    cursor = g.db.cursor()
    cursor.execute('''
        SELECT s.session_id, s.user_id, s.ip_address, s.user_agent,
               s.created_at, s.last_activity, s.expires_at,
               u.username, u.name, r.display_name as role_name
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        JOIN roles r ON u.role_id = r.id
        WHERE s.is_active = 1 AND s.expires_at > datetime('now')
        ORDER BY s.last_activity DESC
    ''')
    active_sessions = [dict(row) for row in cursor.fetchall()]

    # Get recent login activity (last 24h)
    cursor.execute('''
        SELECT al.timestamp, al.username, al.action_type, al.ip_address, al.success
        FROM audit_logs al
        WHERE al.action_type IN ('LOGIN_SUCCESS', 'LOGIN_FAILED', 'LOGOUT')
        AND al.timestamp > datetime('now', '-1 day')
        ORDER BY al.timestamp DESC
        LIMIT 50
    ''')
    login_activity = [dict(row) for row in cursor.fetchall()]

    return render_template('admin/activity.html',
                         user=session.get('name', 'User'),
                         active_sessions=active_sessions,
                         login_activity=login_activity)


@app.route('/admin/sessions/<session_id>/terminate', methods=['POST'])
@login_required
@admin_required
def admin_terminate_session(session_id):
    """Terminate a specific user session."""
    cursor = g.db.cursor()
    cursor.execute('UPDATE sessions SET is_active = 0 WHERE session_id = ?', (session_id,))
    g.db.commit()

    g.audit_logger.log(
        action_type='SESSION_TERMINATED',
        resource_type='session',
        resource_id=session_id,
        success=True
    )

    flash('Session terminated.', 'success')
    return redirect(url_for('admin_activity'))


@app.route('/admin/audit')
@login_required
@admin_required
@audit_action(AuditAction.AUDIT_VIEW, 'audit')
def admin_audit():
    """Audit log viewer."""
    # Get filter parameters
    user_filter = request.args.get('user_id', type=int)
    action_filter = request.args.get('action_type')
    success_filter = request.args.get('success')
    page = request.args.get('page', 1, type=int)
    per_page = 50

    # Convert success filter
    success_bool = None
    if success_filter == 'true':
        success_bool = True
    elif success_filter == 'false':
        success_bool = False

    # Get logs
    offset = (page - 1) * per_page
    logs = get_audit_logs(
        g.db,
        user_id=user_filter,
        action_type=action_filter,
        success=success_bool,
        limit=per_page,
        offset=offset
    )

    # Get summary
    summary = get_audit_summary(g.db, days=7)

    # Get action types for filter dropdown
    action_types = [
        AuditAction.LOGIN_SUCCESS, AuditAction.LOGIN_FAILED, AuditAction.LOGOUT,
        AuditAction.PASSWORD_CHANGE, AuditAction.PASSWORD_RESET,
        AuditAction.USER_CREATE, AuditAction.USER_UPDATE,
        AuditAction.SCREENING_CREATE, AuditAction.SCREENING_VIEW,
        AuditAction.DATA_EXPORT_CSV, AuditAction.DATA_EXPORT_EXCEL
    ]

    return render_template('admin/audit.html',
                         user=session.get('name', 'User'),
                         logs=logs,
                         summary=summary,
                         action_types=action_types,
                         current_filters={
                             'user_id': user_filter,
                             'action_type': action_filter,
                             'success': success_filter
                         },
                         page=page)


# Load model and initialize database on startup
with app.app_context():
    # Initialize database and run migrations
    try:
        conn = get_db_connection()
        init_database(conn)
        migration_result = run_migrations(conn)
        if migration_result['success']:
            logger.info("Database initialized and migrations complete")
        else:
            logger.warning(f"Migration issues: {migration_result}")
        conn.close()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # Load ML model
    load_model()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
