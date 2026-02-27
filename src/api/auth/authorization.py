"""
Authorization Module for ASD Prediction System.
Provides permission-based access control and decorators.
"""

import logging
import sqlite3
from functools import wraps
from typing import List, Union, Optional, Dict

from flask import session, flash, redirect, url_for, g, request

logger = logging.getLogger(__name__)


class Permission:
    """Permission constants for access control."""

    # Screening permissions
    SCREENING_CREATE = 'screening.create'
    SCREENING_VIEW = 'screening.view'
    SCREENING_VIEW_ALL = 'screening.view_all'
    SCREENING_EXPORT = 'screening.export'

    # User management permissions
    USER_CREATE = 'user.create'
    USER_EDIT = 'user.edit'
    USER_VIEW = 'user.view'
    USER_DEACTIVATE = 'user.deactivate'

    # Audit permissions
    AUDIT_VIEW = 'audit.view'

    # Report permissions
    REPORT_GENERATE = 'report.generate'
    REPORT_VIEW = 'report.view'

    # Settings permissions
    SETTINGS_MANAGE = 'settings.manage'


# Cache for user permissions (cleared on permission changes)
_permissions_cache: Dict[int, List[str]] = {}


def clear_permissions_cache(user_id: Optional[int] = None) -> None:
    """
    Clear the permissions cache.

    Args:
        user_id: Specific user ID to clear, or None for all users
    """
    global _permissions_cache
    if user_id is not None:
        _permissions_cache.pop(user_id, None)
    else:
        _permissions_cache.clear()


def get_user_permissions(conn: sqlite3.Connection, user_id: int) -> List[str]:
    """
    Get all permissions for a user based on their role.

    Args:
        conn: Database connection
        user_id: User ID

    Returns:
        List of permission names
    """
    # Check cache
    if user_id in _permissions_cache:
        return _permissions_cache[user_id]

    cursor = conn.execute("""
        SELECT p.name
        FROM permissions p
        JOIN role_permissions rp ON p.id = rp.permission_id
        JOIN users u ON u.role_id = rp.role_id
        WHERE u.id = ? AND u.is_active = 1
    """, (user_id,))

    permissions = [row[0] for row in cursor.fetchall()]

    # Cache the result
    _permissions_cache[user_id] = permissions

    return permissions


def has_permission(
    conn: sqlite3.Connection,
    user_id: int,
    permission: str
) -> bool:
    """
    Check if user has a specific permission.

    Args:
        conn: Database connection
        user_id: User ID
        permission: Permission name to check

    Returns:
        True if user has permission
    """
    permissions = get_user_permissions(conn, user_id)
    return permission in permissions


def has_any_permission(
    conn: sqlite3.Connection,
    user_id: int,
    permissions: List[str]
) -> bool:
    """
    Check if user has any of the specified permissions.

    Args:
        conn: Database connection
        user_id: User ID
        permissions: List of permission names

    Returns:
        True if user has at least one permission
    """
    user_permissions = get_user_permissions(conn, user_id)
    return any(p in user_permissions for p in permissions)


def has_all_permissions(
    conn: sqlite3.Connection,
    user_id: int,
    permissions: List[str]
) -> bool:
    """
    Check if user has all specified permissions.

    Args:
        conn: Database connection
        user_id: User ID
        permissions: List of permission names

    Returns:
        True if user has all permissions
    """
    user_permissions = get_user_permissions(conn, user_id)
    return all(p in user_permissions for p in permissions)


def login_required(f):
    """
    Decorator to require login for route access.

    Redirects to login page if user is not authenticated.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))

        # Check if password change is required
        if session.get('must_change_password'):
            if request.endpoint not in ['change_password_page', 'logout', 'static']:
                flash('You must change your password before continuing.', 'warning')
                return redirect(url_for('change_password_page'))

        return f(*args, **kwargs)
    return decorated_function


def permission_required(
    permission: Union[str, List[str]],
    require_all: bool = False
):
    """
    Decorator to require specific permission(s) for route access.

    Args:
        permission: Permission name or list of permission names
        require_all: If True, require all permissions; if False, require any

    Usage:
        @permission_required(Permission.SCREENING_CREATE)
        def create_screening():
            ...

        @permission_required([Permission.USER_VIEW, Permission.USER_EDIT], require_all=True)
        def edit_user():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check login
            if 'user_id' not in session:
                flash('Please log in to access this page.', 'warning')
                return redirect(url_for('login'))

            user_id = session.get('user_id')
            conn = g.db

            # Normalize to list
            permissions = [permission] if isinstance(permission, str) else permission

            # Check permissions
            if require_all:
                has_access = has_all_permissions(conn, user_id, permissions)
            else:
                has_access = has_any_permission(conn, user_id, permissions)

            if not has_access:
                logger.warning(
                    f"Access denied for user {user_id} to {request.endpoint}. "
                    f"Required: {permissions}"
                )
                flash('You do not have permission to access this resource.', 'danger')
                return redirect(url_for('dashboard'))

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def admin_required(f):
    """
    Decorator to require admin role for route access.

    Shortcut for permission_required with all admin permissions.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))

        if session.get('role') != 'admin':
            logger.warning(
                f"Admin access denied for user {session.get('user_id')} "
                f"(role: {session.get('role')}) to {request.endpoint}"
            )
            flash('Administrator access required.', 'danger')
            return redirect(url_for('dashboard'))

        return f(*args, **kwargs)
    return decorated_function


def can_view_all_screenings(conn: sqlite3.Connection, user_id: int) -> bool:
    """
    Check if user can view all screenings (not just their own).

    Args:
        conn: Database connection
        user_id: User ID

    Returns:
        True if user can view all screenings
    """
    return has_permission(conn, user_id, Permission.SCREENING_VIEW_ALL)


def can_export_data(conn: sqlite3.Connection, user_id: int) -> bool:
    """
    Check if user can export screening data.

    Args:
        conn: Database connection
        user_id: User ID

    Returns:
        True if user can export data
    """
    return has_permission(conn, user_id, Permission.SCREENING_EXPORT)


def can_manage_users(conn: sqlite3.Connection, user_id: int) -> bool:
    """
    Check if user can manage other users.

    Args:
        conn: Database connection
        user_id: User ID

    Returns:
        True if user can manage users
    """
    return has_any_permission(conn, user_id, [
        Permission.USER_CREATE,
        Permission.USER_EDIT,
        Permission.USER_DEACTIVATE
    ])


def get_role_permissions(conn: sqlite3.Connection, role_name: str) -> List[str]:
    """
    Get all permissions for a specific role.

    Args:
        conn: Database connection
        role_name: Role name

    Returns:
        List of permission names
    """
    cursor = conn.execute("""
        SELECT p.name
        FROM permissions p
        JOIN role_permissions rp ON p.id = rp.permission_id
        JOIN roles r ON r.id = rp.role_id
        WHERE r.name = ?
    """, (role_name,))

    return [row[0] for row in cursor.fetchall()]


def check_resource_access(
    conn: sqlite3.Connection,
    user_id: int,
    resource_type: str,
    resource_owner_id: Optional[int] = None
) -> bool:
    """
    Check if user can access a specific resource.

    For resources with owners, checks if user is owner or has view_all permission.

    Args:
        conn: Database connection
        user_id: User ID requesting access
        resource_type: Type of resource (e.g., 'screening')
        resource_owner_id: Owner user ID of the resource

    Returns:
        True if access is allowed
    """
    # If user is owner, allow access
    if resource_owner_id is not None and user_id == resource_owner_id:
        return True

    # Check view_all permission for the resource type
    view_all_permission = f"{resource_type}.view_all"
    return has_permission(conn, user_id, view_all_permission)
