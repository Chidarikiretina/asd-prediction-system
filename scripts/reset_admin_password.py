#!/usr/bin/env python3
"""
Reset the admin user password directly in the database.

Usage (run from project root):
    python scripts/reset_admin_password.py
"""

import sys
import os
import getpass
import sqlite3
from pathlib import Path


def get_db_path() -> Path:
    env_path = os.getenv('ASD_DB_PATH')
    if env_path:
        return Path(env_path)
    # Default: project_root/data/asd_system.db
    return Path(__file__).parent.parent / 'data' / 'asd_system.db'


def hash_password(password: str) -> str:
    try:
        import bcrypt
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(12))
        return hashed.decode('utf-8')
    except ImportError:
        import hashlib
        print("WARNING: bcrypt not installed — using SHA-256 fallback (less secure).")
        return hashlib.sha256(password.encode('utf-8')).hexdigest()


def main():
    db_path = get_db_path()

    print("=" * 50)
    print("  ASD System — Admin Password Reset")
    print("=" * 50)
    print(f"Database: {db_path}")
    print()

    if not db_path.exists():
        print("ERROR: Database file not found.")
        print("Start the app at least once to create the database, then retry.")
        sys.exit(1)

    new_password = getpass.getpass("New admin password: ")
    if not new_password:
        print("ERROR: Password cannot be empty.")
        sys.exit(1)
    if len(new_password) < 8:
        print("ERROR: Password must be at least 8 characters.")
        sys.exit(1)

    confirm = getpass.getpass("Confirm new password: ")
    if new_password != confirm:
        print("ERROR: Passwords do not match.")
        sys.exit(1)

    password_hash = hash_password(new_password)

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()
        if not row:
            print("ERROR: Admin user not found in the database.")
            sys.exit(1)

        conn.execute("""
            UPDATE users SET
                password_hash       = ?,
                failed_attempts     = 0,
                locked_until        = NULL,
                must_change_password = 1,
                updated_at          = CURRENT_TIMESTAMP
            WHERE username = 'admin'
        """, (password_hash,))
        conn.commit()
    finally:
        conn.close()

    print()
    print("Admin password reset successfully.")
    print("The admin will be prompted to change it on next login.")


if __name__ == '__main__':
    main()
