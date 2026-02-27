"""
Authentication, Authorization, and Accountability (AAA) module.
Provides comprehensive security features for the ASD Prediction System.
"""

from .password_security import PasswordHasher, PasswordPolicy, AccountLockout
from .authentication import authenticate_user, create_session, invalidate_session
from .authorization import (
    Permission, login_required, permission_required, admin_required,
    has_permission, get_user_permissions
)
from .audit import AuditLogger, AuditAction, audit_action
from .user_management import UserManager

__all__ = [
    # Password security
    'PasswordHasher', 'PasswordPolicy', 'AccountLockout',

    # Authentication
    'authenticate_user', 'create_session', 'invalidate_session',

    # Authorization
    'Permission', 'login_required', 'permission_required', 'admin_required',
    'has_permission', 'get_user_permissions',

    # Audit
    'AuditLogger', 'AuditAction', 'audit_action',

    # User management
    'UserManager'
]
