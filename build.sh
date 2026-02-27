#!/usr/bin/env bash
# =============================================================
# Render Build Script - ASD Prediction System
# Runs once during deployment build phase
# =============================================================
set -e  # Exit on any error

echo "=== ASD Prediction System - Build Starting ==="

# Install Python dependencies
echo "[1/3] Installing dependencies..."
pip install -r requirements/requirements.txt

# Ensure required directories exist
echo "[2/3] Setting up directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p src/api/static/img

# Generate synthetic training data if it doesn't exist
echo "[3/3] Checking training data..."
if [ ! -f "data/raw/asd_train_data.csv" ]; then
    echo "  Training data not found - generating synthetic data..."
    python scripts/data/generate_synthetic_data.py
    echo "  Training data generated."
else
    echo "  Training data found - skipping generation."
fi

echo "=== Build Complete ==="
