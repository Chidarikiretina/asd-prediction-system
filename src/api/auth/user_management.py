"""
User Management Module for ASD Prediction System.
Provides admin functionality for user CRUD operations.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from .password_security import PasswordHasher, PasswordPolicy, generate_temporary_password

logger = logging.getLogger(__name__)


class UserManager:
    """
    User management operations for administrators.
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize user manager.

        Args:
            conn: Database connection
        """
        self.conn = conn
        self.hasher = PasswordHasher()
        self.policy = PasswordPolicy()

    def list_users(
        self,
        include_inactive: bool = False,
        role_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Get list of all users.

        Args:
            include_inactive: Include deactivated users
            role_filter: Filter by role name

        Returns:
            List of user dictionaries
        """
        query = """
            SELECT
                u.id, u.username, u.name, u.email,
                r.name as role, r.display_name as role_display,
                u.facility, u.is_active, u.last_login, u.created_at,
                u.failed_login_attempts, u.locked_until
            FROM users u
            JOIN roles r ON u.role_id = r.id
            WHERE 1=1
        """
        params = []

        if not include_inactive:
            query += " AND u.is_active = 1"

        if role_filter:
            query += " AND r.name = ?"
            params.append(role_filter)

        query += " ORDER BY u.name"

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_user(self, user_id: int) -> Optional[Dict]:
        """
        Get single user by ID.

        Args:
            user_id: User ID

        Returns:
            User dictionary or None
        """
        cursor = self.conn.execute("""
            SELECT
                u.*, r.name as role_name, r.display_name as role_display
            FROM users u
            JOIN roles r ON u.role_id = r.id
            WHERE u.id = ?
        """, (user_id,))

        row = cursor.fetchone()
        if row:
            user = dict(row)
            # Remove sensitive data
            del user['password_hash']
            return user
        return None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """
        Get user by username.

        Args:
            username: Username

        Returns:
            User dictionary or None
        """
        cursor = self.conn.execute("""
            SELECT u.id FROM users u WHERE u.username = ?
        """, (username,))

        row = cursor.fetchone()
        if row:
            return self.get_user(row['id'])
        return None

    def create_user(
        self,
        username: str,
        password: str,
        name: str,
        role_name: str,
        facility: str,
        email: Optional[str] = None,
        created_by: Optional[int] = None,
        must_change_password: bool = True
    ) -> Tuple[bool, str, Optional[int]]:
        """
        Create a new user.

        Args:
            username: Unique username
            password: Initial password
            name: Full name
            role_name: Role name
            facility: Facility name
            email: Email address (optional)
            created_by: ID of user creating this user
            must_change_password: Require password change on first login

        Returns:
            Tuple of (success, message, user_id)
        """
        # Validate username format
        if not username or len(username) < 3:
            return (False, "Username must be at least 3 characters", None)

        if not username.isalnum() and '_' not in username:
            return (False, "Username can only contain letters, numbers, and underscores", None)

        # Check username uniqueness
        cursor = self.conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (username.lower(),)
        )
        if cursor.fetchone():
            return (False, "Username already exists", None)

        # Validate password
        is_valid, errors = self.policy.validate(password)
        if not is_valid:
            return (False, "; ".join(errors), None)

        # Get role ID
        cursor = self.conn.execute(
            "SELECT id FROM roles WHERE name = ?",
            (role_name,)
        )
        role_row = cursor.fetchone()
        if not role_row:
            return (False, f"Invalid role: {role_name}", None)
        role_id = role_row['id']

        # Hash password
        password_hash = self.hasher.hash_password(password)

        try:
            # Insert user
            cursor = self.conn.execute("""
                INSERT INTO users (
                    username, password_hash, name, email, role_id, facility,
                    must_change_password, password_changed_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                username.lower(),
                password_hash,
                name,
                email,
                role_id,
                facility,
                must_change_password,
                datetime.utcnow().isoformat(),
                created_by
            ))
            self.conn.commit()

            user_id = cursor.lastrowid
            logger.info(f"Created user '{username}' with ID {user_id}")
            return (True, "User created successfully", user_id)

        except Exception as e:
            logger.error(f"Failed to create user '{username}': {e}")
            self.conn.rollback()
            return (False, f"Failed to create user: {str(e)}", None)

    def update_user(
        self,
        user_id: int,
        name: Optional[str] = None,
        email: Optional[str] = None,
        role_name: Optional[str] = None,
        facility: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Update user details.

        Args:
            user_id: User ID
            name: New name (optional)
            email: New email (optional)
            role_name: New role (optional)
            facility: New facility (optional)

        Returns:
            Tuple of (success, message)
        """
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if email is not None:
            updates.append("email = ?")
            params.append(email)

        if facility is not None:
            updates.append("facility = ?")
            params.append(facility)

        if role_name is not None:
            cursor = self.conn.execute(
                "SELECT id FROM roles WHERE name = ?",
                (role_name,)
            )
            role_row = cursor.fetchone()
            if not role_row:
                return (False, f"Invalid role: {role_name}")
            updates.append("role_id = ?")
            params.append(role_row['id'])

        if not updates:
            return (False, "No updates provided")

        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(user_id)

        try:
            self.conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
                params
            )
            self.conn.commit()
            logger.info(f"Updated user {user_id}")
            return (True, "User updated successfully")
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            self.conn.rollback()
            return (False, f"Failed to update user: {str(e)}")

    def reset_password(
        self,
        user_id: int,
        new_password: Optional[str] = None,
        must_change: bool = True
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Reset user password.

        Args:
            user_id: User ID
            new_password: New password (generates random if not provided)
            must_change: Require password change on next login

        Returns:
            Tuple of (success, message, temporary_password if generated)
        """
        temp_password = None

        if new_password is None:
            # Generate temporary password
            new_password = generate_temporary_password()
            temp_password = new_password

        # Validate new password
        is_valid, errors = self.policy.validate(new_password)
        if not is_valid:
            return (False, "; ".join(errors), None)

        password_hash = self.hasher.hash_password(new_password)

        try:
            self.conn.execute("""
                UPDATE users SET
                    password_hash = ?,
                    password_changed_at = ?,
                    must_change_password = ?,
                    failed_login_attempts = 0,
                    locked_until = NULL,
                    updated_at = ?
                WHERE id = ?
            """, (
                password_hash,
                datetime.utcnow().isoformat(),
                must_change,
                datetime.utcnow().isoformat(),
                user_id
            ))
            self.conn.commit()
            logger.info(f"Reset password for user {user_id}")
            return (True, "Password reset successfully", temp_password)
        except Exception as e:
            logger.error(f"Failed to reset password for user {user_id}: {e}")
            self.conn.rollback()
            return (False, f"Failed to reset password: {str(e)}", None)

    def deactivate_user(self, user_id: int) -> Tuple[bool, str]:
        """
        Deactivate a user account.

        Args:
            user_id: User ID

        Returns:
            Tuple of (success, message)
        """
        try:
            self.conn.execute("""
                UPDATE users SET
                    is_active = 0,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), user_id))
            self.conn.commit()
            logger.info(f"Deactivated user {user_id}")
            return (True, "User deactivated successfully")
        except Exception as e:
            logger.error(f"Failed to deactivate user {user_id}: {e}")
            self.conn.rollback()
            return (False, f"Failed to deactivate user: {str(e)}")

    def reactivate_user(self, user_id: int) -> Tuple[bool, str]:
        """
        Reactivate a deactivated user account.

        Args:
            user_id: User ID

        Returns:
            Tuple of (success, message)
        """
        try:
            self.conn.execute("""
                UPDATE users SET
                    is_active = 1,
                    failed_login_attempts = 0,
                    locked_until = NULL,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), user_id))
            self.conn.commit()
            logger.info(f"Reactivated user {user_id}")
            return (True, "User reactivated successfully")
        except Exception as e:
            logger.error(f"Failed to reactivate user {user_id}: {e}")
            self.conn.rollback()
            return (False, f"Failed to reactivate user: {str(e)}")

    def unlock_account(self, user_id: int) -> Tuple[bool, str]:
        """
        Unlock a locked user account.

        Args:
            user_id: User ID

        Returns:
            Tuple of (success, message)
        """
        try:
            self.conn.execute("""
                UPDATE users SET
                    failed_login_attempts = 0,
                    locked_until = NULL,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), user_id))
            self.conn.commit()
            logger.info(f"Unlocked account for user {user_id}")
            return (True, "Account unlocked successfully")
        except Exception as e:
            logger.error(f"Failed to unlock account for user {user_id}: {e}")
            self.conn.rollback()
            return (False, f"Failed to unlock account: {str(e)}")

    def get_user_activity(
        self,
        user_id: int,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get user's recent activity from audit logs.

        Args:
            user_id: User ID
            limit: Maximum number of entries

        Returns:
            List of audit log entries
        """
        cursor = self.conn.execute("""
            SELECT
                id, timestamp, action_type, resource_type, resource_id,
                ip_address, success, details
            FROM audit_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))

        return [dict(row) for row in cursor.fetchall()]

    def get_roles(self) -> List[Dict]:
        """
        Get all available roles.

        Returns:
            List of role dictionaries
        """
        cursor = self.conn.execute("""
            SELECT id, name, display_name, description
            FROM roles
            WHERE is_active = 1
            ORDER BY id
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_user_statistics(self) -> Dict:
        """
        Get user statistics summary.

        Returns:
            Dictionary with user statistics
        """
        # Total users
        cursor = self.conn.execute("SELECT COUNT(*) FROM users")
        total = cursor.fetchone()[0]

        # Active users
        cursor = self.conn.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        active = cursor.fetchone()[0]

        # Users by role
        cursor = self.conn.execute("""
            SELECT r.display_name, COUNT(u.id) as count
            FROM roles r
            LEFT JOIN users u ON r.id = u.role_id AND u.is_active = 1
            GROUP BY r.id
            ORDER BY r.id
        """)
        by_role = {row['display_name']: row['count'] for row in cursor.fetchall()}

        # Locked accounts
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM users
            WHERE locked_until IS NOT NULL AND locked_until > datetime('now')
        """)
        locked = cursor.fetchone()[0]

        # Recently active (last 24 hours)
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM users
            WHERE last_login > datetime('now', '-1 day')
        """)
        recently_active = cursor.fetchone()[0]

        return {
            'total_users': total,
            'active_users': active,
            'inactive_users': total - active,
            'locked_accounts': locked,
            'recently_active': recently_active,
            'users_by_role': by_role
        }
