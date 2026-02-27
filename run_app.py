"""
Run the ASD Prediction System Web Application

Usage: python run_app.py
Then open http://localhost:5000 in your browser
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    # Check for Flask
    try:
        from flask import Flask
    except ImportError:
        print("Flask is not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
        print("Flask installed successfully!\n")

    # Import and run the app
    from api.app import app

    print("=" * 60)
    print("  ASD PREDICTION SYSTEM - WEB APPLICATION")
    print("=" * 60)
    print("\n  Starting server...")
    print("\n  Open your browser and go to:")
    print("  --> http://localhost:5050")
    print("\n  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    app.run(debug=True, host='127.0.0.1', port=5050)


if __name__ == '__main__':
    main()
