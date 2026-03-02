"""
Database Migrations for ASD Prediction System.
Handles schema initialization and migration of legacy users.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .db import get_db_connection, init_database

logger = logging.getLogger(__name__)

# Legacy users from the original in-memory USERS dictionary
# These will be migrated to the database on first run
LEGACY_USERS = {
    'admin': {
        'password': 'admin123',
        'name': 'System Administrator',
        'role': 'admin',
        'facility': 'Central Health Office',
        'email': 'admin@health.gov.zw'
    },
    'drmoyo': {
        'password': 'moyo2024',
        'name': 'Dr. Tendai Moyo',
        'role': 'pediatrician',
        'facility': 'Parirenyatwa Hospital',
        'email': 'drmoyo@parirenyatwa.co.zw'
    },
    'nursechipo': {
        'password': 'chipo2024',
        'name': 'Sister Chipo Ndlovu',
        'role': 'nurse',
        'facility': 'Harare Central Clinic',
        'email': 'nursechipo@harareclinic.co.zw'
    },
    'healthworker': {
        'password': 'asd2024',
        'name': 'Community Health Worker',
        'role': 'chw',
        'facility': 'Chitungwiza District',
        'email': 'chw@chitungwiza.gov.zw'
    }
}


def get_role_id(conn: sqlite3.Connection, role_name: str) -> Optional[int]:
    """Get role ID by name."""
    cursor = conn.execute(
        "SELECT id FROM roles WHERE name = ?",
        (role_name,)
    )
    row = cursor.fetchone()
    return row[0] if row else None


def user_exists(conn: sqlite3.Connection, username: str) -> bool:
    """Check if user already exists."""
    cursor = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        (username,)
    )
    return cursor.fetchone() is not None


def migrate_legacy_users(conn: Optional[sqlite3.Connection] = None) -> Dict:
    """
    Migrate legacy in-memory users to database.

    Users are created with must_change_password=True to force
    them to set a new secure password on first login.

    Returns:
        Dict with migration status and count
    """
    from auth.password_security import PasswordHasher

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    hasher = PasswordHasher()
    migrated = 0
    skipped = 0
    errors = []

    try:
        for username, data in LEGACY_USERS.items():
            try:
                # Skip if user already exists
                if user_exists(conn, username):
                    logger.info(f"User '{username}' already exists, skipping")
                    skipped += 1
                    continue

                # Get role ID
                role_id = get_role_id(conn, data['role'])
                if not role_id:
                    logger.error(f"Role '{data['role']}' not found for user '{username}'")
                    errors.append(f"Role not found: {data['role']}")
                    continue

                # Hash password with bcrypt
                password_hash = hasher.hash_password(data['password'])

                # Insert user with must_change_password = True
                conn.execute("""
                    INSERT INTO users (
                        username, password_hash, name, email, role_id, facility,
                        must_change_password, password_changed_at, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, 1)
                """, (
                    username,
                    password_hash,
                    data['name'],
                    data.get('email'),
                    role_id,
                    data['facility'],
                    datetime.utcnow()
                ))

                migrated += 1
                logger.info(f"Migrated user '{username}' with role '{data['role']}'")

            except Exception as e:
                logger.error(f"Failed to migrate user '{username}': {e}")
                errors.append(f"{username}: {str(e)}")

        conn.commit()

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        if close_conn:
            conn.close()

    result = {
        'success': len(errors) == 0,
        'migrated': migrated,
        'skipped': skipped,
        'errors': errors
    }

    logger.info(f"Migration complete: {migrated} migrated, {skipped} skipped, {len(errors)} errors")
    return result


def fix_role_permissions(conn: sqlite3.Connection) -> Dict:
    """
    Enforce Separation of Duties by correcting role permissions.

    - Removes clinical (screening/report) permissions from admin role.
    - Adds report.generate to nurse role.
    - Safe to run multiple times (idempotent).
    """
    result = {'fixed': False, 'changes': [], 'errors': []}
    try:
        # --- Admin: remove screening and report permissions (SoD) ---
        admin_clinical = [
            'screening.create', 'screening.view', 'screening.view_all',
            'screening.export', 'report.generate', 'report.view'
        ]
        placeholders = ','.join('?' * len(admin_clinical))
        cursor = conn.execute(f"""
            DELETE FROM role_permissions
            WHERE role_id = (SELECT id FROM roles WHERE name = 'admin')
              AND permission_id IN (
                  SELECT id FROM permissions WHERE name IN ({placeholders})
              )
        """, admin_clinical)
        if cursor.rowcount > 0:
            result['changes'].append(f"Removed {cursor.rowcount} clinical permission(s) from admin role")

        # --- Nurse: add report.generate if missing ---
        conn.execute("""
            INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
            SELECT r.id, p.id FROM roles r, permissions p
            WHERE r.name = 'nurse' AND p.name = 'report.generate'
        """)

        conn.commit()
        result['fixed'] = True
        logger.info("Role permissions fixed for Separation of Duties")
    except Exception as e:
        logger.error(f"fix_role_permissions failed: {e}")
        result['errors'].append(str(e))
        conn.rollback()

    # Clear the permissions cache so users see updated permissions immediately
    try:
        from auth.authorization import clear_permissions_cache
        clear_permissions_cache()
    except Exception:
        pass

    return result


def run_migrations(conn: Optional[sqlite3.Connection] = None) -> Dict:
    """
    Run all pending migrations.

    This includes:
    1. Initializing the database schema
    2. Migrating legacy users

    Returns:
        Dict with overall migration status
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    results = {
        'schema_initialized': False,
        'users_migration': None,
        'success': False
    }

    try:
        # Initialize schema (idempotent - safe to run multiple times)
        init_database(conn)
        results['schema_initialized'] = True
        logger.info("Database schema initialized/verified")

        # Migrate legacy users
        user_migration = migrate_legacy_users(conn)
        results['users_migration'] = user_migration

        # Enforce Separation of Duties — fix role permissions on existing databases
        sod_result = fix_role_permissions(conn)
        results['sod_migration'] = sod_result

        results['success'] = user_migration['success'] and sod_result['fixed']

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        results['error'] = str(e)
    finally:
        if close_conn:
            conn.close()

    return results


def check_migration_status(conn: Optional[sqlite3.Connection] = None) -> Dict:
    """
    Check current migration status.

    Returns:
        Dict with database status information
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        # Check if tables exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        # Check user count
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

        # Check role count
        role_count = conn.execute("SELECT COUNT(*) FROM roles").fetchone()[0]

        # Check permission count
        permission_count = conn.execute("SELECT COUNT(*) FROM permissions").fetchone()[0]

        # Check which legacy users exist
        legacy_status = {}
        for username in LEGACY_USERS.keys():
            legacy_status[username] = user_exists(conn, username)

        return {
            'tables': tables,
            'user_count': user_count,
            'role_count': role_count,
            'permission_count': permission_count,
            'legacy_users_migrated': legacy_status,
            'all_migrated': all(legacy_status.values())
        }

    finally:
        if close_conn:
            conn.close()


def reset_database(conn: Optional[sqlite3.Connection] = None) -> bool:
    """
    Reset database by dropping all tables and re-initializing.
    WARNING: This will delete all data!

    Returns:
        True if successful
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        # Get all tables
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [row[0] for row in cursor.fetchall()]

        # Drop all tables
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"Dropped table: {table}")

        conn.commit()

        # Re-initialize
        init_database(conn)

        # Run migrations
        run_migrations(conn)

        logger.info("Database reset complete")
        return True

    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        conn.rollback()
        return False
    finally:
        if close_conn:
            conn.close()
