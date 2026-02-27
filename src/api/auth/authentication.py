"""
Authentication Module for ASD Prediction System.
Handles user login, logout, and session management.
"""

import logging
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict

from .password_security import PasswordHasher, AccountLockout

logger = logging.getLogger(__name__)

# Session configuration
SESSION_LIFETIME_HOURS = 8


def authenticate_user(
    conn: sqlite3.Connection,
    username: str,
    password: str
) -> Tuple[bool, Optional[Dict], str]:
    """
    Authenticate user with username and password.

    Handles:
    - Password verification
    - Account lockout checks
    - Failed attempt tracking
    - Last login updates

    Args:
        conn: Database connection
        username: Username to authenticate
        password: Password to verify

    Returns:
        Tuple of (success, user_dict, message)
    """
    hasher = PasswordHasher()
    lockout = AccountLockout()

    # Fetch user from database
    cursor = conn.execute("""
        SELECT u.*, r.name as role_name, r.display_name as role_display
        FROM users u
        JOIN roles r ON u.role_id = r.id
        WHERE u.username = ?
    """, (username,))

    row = cursor.fetchone()

    if not row:
        logger.warning(f"Login attempt for non-existent user: {username}")
        return (False, None, "Invalid username or password")

    user = dict(row)

    # Check if account is active
    if not user['is_active']:
        logger.warning(f"Login attempt for deactivated user: {username}")
        return (False, None, "Account has been deactivated. Contact administrator.")

    # Check account lockout
    locked_until = None
    if user['locked_until']:
        locked_until = datetime.fromisoformat(user['locked_until'])

    is_locked, minutes_remaining = lockout.check_lockout(
        user['failed_login_attempts'],
        locked_until
    )

    if is_locked:
        logger.warning(f"Login attempt for locked account: {username}")
        return (False, None, f"Account is locked. Try again in {minutes_remaining} minutes.")

    # Verify password
    if not hasher.verify_password(password, user['password_hash']):
        # Increment failed attempts
        new_attempts = user['failed_login_attempts'] + 1

        # Check if should lock
        if lockout.should_lock(new_attempts):
            lock_until = lockout.get_lockout_until()
            conn.execute("""
                UPDATE users SET
                    failed_login_attempts = ?,
                    locked_until = ?
                WHERE id = ?
            """, (new_attempts, lock_until.isoformat(), user['id']))
            conn.commit()
            logger.warning(f"Account locked due to failed attempts: {username}")
            return (False, None, f"Account locked due to too many failed attempts. Try again in {lockout.duration.seconds // 60} minutes.")
        else:
            conn.execute("""
                UPDATE users SET failed_login_attempts = ? WHERE id = ?
            """, (new_attempts, user['id']))
            conn.commit()
            remaining = lockout.get_attempts_remaining(new_attempts)
            logger.warning(f"Failed login attempt for user: {username} ({remaining} attempts remaining)")
            return (False, None, f"Invalid username or password. {remaining} attempts remaining.")

    # Successful authentication
    # Reset failed attempts and update last login
    conn.execute("""
        UPDATE users SET
            failed_login_attempts = 0,
            locked_until = NULL,
            last_login = ?
        WHERE id = ?
    """, (datetime.utcnow().isoformat(), user['id']))
    conn.commit()

    # Check if password hash needs upgrade (SHA-256 to bcrypt)
    if hasher.needs_rehash(user['password_hash']):
        new_hash = hasher.hash_password(password)
        conn.execute("""
            UPDATE users SET password_hash = ? WHERE id = ?
        """, (new_hash, user['id']))
        conn.commit()
        logger.info(f"Upgraded password hash for user: {username}")

    logger.info(f"User authenticated successfully: {username}")

    # Return user info (without sensitive data)
    return (True, {
        'id': user['id'],
        'username': user['username'],
        'name': user['name'],
        'email': user['email'],
        'role_id': user['role_id'],
        'role_name': user['role_name'],
        'role_display': user['role_display'],
        'facility': user['facility'],
        'must_change_password': bool(user['must_change_password']),
        'last_login': user['last_login']
    }, "Login successful")


def create_session(
    conn: sqlite3.Connection,
    user_id: int,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> str:
    """
    Create a new session for authenticated user.

    Args:
        conn: Database connection
        user_id: User ID
        ip_address: Client IP address
        user_agent: Client user agent string

    Returns:
        Session ID string
    """
    session_id = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=SESSION_LIFETIME_HOURS)

    conn.execute("""
        INSERT INTO sessions (session_id, user_id, ip_address, user_agent, expires_at)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, user_id, ip_address, user_agent, expires_at.isoformat()))
    conn.commit()

    logger.debug(f"Created session for user {user_id}")
    return session_id


def validate_session(
    conn: sqlite3.Connection,
    session_id: str
) -> Optional[Dict]:
    """
    Validate session and return user info if valid.

    Args:
        conn: Database connection
        session_id: Session ID to validate

    Returns:
        User dict if session is valid, None otherwise
    """
    cursor = conn.execute("""
        SELECT s.*, u.username, u.name, u.role_id, r.name as role_name
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        JOIN roles r ON u.role_id = r.id
        WHERE s.session_id = ?
          AND s.is_active = 1
          AND s.expires_at > ?
          AND u.is_active = 1
    """, (session_id, datetime.utcnow().isoformat()))

    row = cursor.fetchone()
    if not row:
        return None

    # Update last activity
    conn.execute("""
        UPDATE sessions SET last_activity = ? WHERE session_id = ?
    """, (datetime.utcnow().isoformat(), session_id))
    conn.commit()

    return dict(row)


def invalidate_session(
    conn: sqlite3.Connection,
    session_id: str
) -> bool:
    """
    Invalidate (logout) a session.

    Args:
        conn: Database connection
        session_id: Session ID to invalidate

    Returns:
        True if session was invalidated
    """
    cursor = conn.execute("""
        UPDATE sessions SET is_active = 0 WHERE session_id = ?
    """, (session_id,))
    conn.commit()

    if cursor.rowcount > 0:
        logger.debug(f"Invalidated session: {session_id[:8]}...")
        return True
    return False


def invalidate_user_sessions(
    conn: sqlite3.Connection,
    user_id: int
) -> int:
    """
    Invalidate all sessions for a user (e.g., after password change).

    Args:
        conn: Database connection
        user_id: User ID

    Returns:
        Number of sessions invalidated
    """
    cursor = conn.execute("""
        UPDATE sessions SET is_active = 0
        WHERE user_id = ? AND is_active = 1
    """, (user_id,))
    conn.commit()

    count = cursor.rowcount
    if count > 0:
        logger.info(f"Invalidated {count} sessions for user {user_id}")
    return count


def cleanup_expired_sessions(conn: sqlite3.Connection) -> int:
    """
    Remove expired sessions from database.

    Args:
        conn: Database connection

    Returns:
        Number of sessions cleaned up
    """
    cursor = conn.execute("""
        DELETE FROM sessions WHERE expires_at < ?
    """, (datetime.utcnow().isoformat(),))
    conn.commit()

    count = cursor.rowcount
    if count > 0:
        logger.info(f"Cleaned up {count} expired sessions")
    return count


def get_active_sessions(
    conn: sqlite3.Connection,
    user_id: int
) -> list:
    """
    Get all active sessions for a user.

    Args:
        conn: Database connection
        user_id: User ID

    Returns:
        List of session dictionaries
    """
    cursor = conn.execute("""
        SELECT id, session_id, ip_address, user_agent, created_at, last_activity
        FROM sessions
        WHERE user_id = ?
          AND is_active = 1
          AND expires_at > ?
        ORDER BY last_activity DESC
    """, (user_id, datetime.utcnow().isoformat()))

    return [dict(row) for row in cursor.fetchall()]


def change_password(
    conn: sqlite3.Connection,
    user_id: int,
    current_password: str,
    new_password: str
) -> Tuple[bool, str]:
    """
    Change user password.

    Args:
        conn: Database connection
        user_id: User ID
        current_password: Current password for verification
        new_password: New password to set

    Returns:
        Tuple of (success, message)
    """
    from .password_security import PasswordPolicy

    hasher = PasswordHasher()
    policy = PasswordPolicy()

    # Get current password hash
    cursor = conn.execute(
        "SELECT password_hash FROM users WHERE id = ?",
        (user_id,)
    )
    row = cursor.fetchone()

    if not row:
        return (False, "User not found")

    # Verify current password
    if not hasher.verify_password(current_password, row['password_hash']):
        return (False, "Current password is incorrect")

    # Validate new password
    is_valid, errors = policy.validate(new_password)
    if not is_valid:
        return (False, "; ".join(errors))

    # Check password isn't same as current
    if hasher.verify_password(new_password, row['password_hash']):
        return (False, "New password must be different from current password")

    # Hash and save new password
    new_hash = hasher.hash_password(new_password)

    conn.execute("""
        UPDATE users SET
            password_hash = ?,
            password_changed_at = ?,
            must_change_password = 0
        WHERE id = ?
    """, (new_hash, datetime.utcnow().isoformat(), user_id))

    # Save to password history (optional, for preventing reuse)
    conn.execute("""
        INSERT INTO password_history (user_id, password_hash)
        VALUES (?, ?)
    """, (user_id, new_hash))

    conn.commit()

    # Invalidate other sessions (security measure)
    invalidate_user_sessions(conn, user_id)

    logger.info(f"Password changed for user {user_id}")
    return (True, "Password changed successfully")
