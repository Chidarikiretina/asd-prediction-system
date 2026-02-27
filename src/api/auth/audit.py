"""
Audit Logging Module for ASD Prediction System.
Provides comprehensive action logging for accountability.
"""

import json
import logging
import sqlite3
from functools import wraps
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable

from flask import request, session, g

logger = logging.getLogger(__name__)


class AuditAction:
    """Audit action type constants."""

    # Authentication events
    LOGIN_SUCCESS = 'login_success'
    LOGIN_FAILED = 'login_failed'
    LOGOUT = 'logout'
    PASSWORD_CHANGE = 'password_change'
    PASSWORD_RESET = 'password_reset'
    ACCOUNT_LOCKED = 'account_locked'
    ACCOUNT_UNLOCKED = 'account_unlocked'

    # User management events
    USER_CREATE = 'user_create'
    USER_UPDATE = 'user_update'
    USER_DEACTIVATE = 'user_deactivate'
    USER_REACTIVATE = 'user_reactivate'
    USER_VIEW = 'user_view'

    # Screening events
    SCREENING_CREATE = 'screening_create'
    SCREENING_VIEW = 'screening_view'
    SCREENING_UPDATE = 'screening_update'
    SCREENING_DELETE = 'screening_delete'

    # Report events
    REPORT_VIEW = 'report_view'
    REPORT_GENERATE = 'report_generate'
    REPORT_PRINT = 'report_print'

    # Export events
    DATA_EXPORT_CSV = 'data_export_csv'
    DATA_EXPORT_EXCEL = 'data_export_excel'

    # Admin events
    SETTINGS_VIEW = 'settings_view'
    SETTINGS_CHANGE = 'settings_change'
    AUDIT_VIEW = 'audit_view'

    # System events
    SYSTEM_STARTUP = 'system_startup'
    SYSTEM_ERROR = 'system_error'
    DATABASE_MIGRATION = 'database_migration'


class AuditLogger:
    """
    Audit logging utility for recording user actions.
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize audit logger.

        Args:
            conn: Database connection
        """
        self.conn = conn

    def log(
        self,
        action_type: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        user_id: Optional[int] = None,
        username: Optional[str] = None
    ) -> int:
        """
        Log an audit event.

        Args:
            action_type: Type of action (from AuditAction)
            resource_type: Type of resource affected
            resource_id: ID of the resource affected
            details: Additional details as dict
            success: Whether the action succeeded
            error_message: Error message if failed
            user_id: Override user ID (uses session by default)
            username: Override username (uses session by default)

        Returns:
            ID of the created audit log entry
        """
        # Get user info from session if not provided
        if user_id is None:
            user_id = session.get('user_id')
        if username is None:
            username = session.get('user')

        session_id = session.get('session_id')

        # Get request info if available
        ip_address = None
        user_agent = None
        request_method = None
        request_path = None

        if request:
            ip_address = request.remote_addr
            user_agent = request.user_agent.string if request.user_agent else None
            request_method = request.method
            request_path = request.path

        # Serialize details to JSON
        details_json = json.dumps(details) if details else None

        try:
            cursor = self.conn.execute("""
                INSERT INTO audit_logs (
                    user_id, username, session_id, action_type,
                    resource_type, resource_id, ip_address, user_agent,
                    request_method, request_path, details, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                username,
                session_id,
                action_type,
                resource_type,
                str(resource_id) if resource_id else None,
                ip_address,
                user_agent,
                request_method,
                request_path,
                details_json,
                success,
                error_message
            ))
            self.conn.commit()

            log_id = cursor.lastrowid
            logger.debug(
                f"Audit log created: {action_type} by {username} "
                f"(success={success}, id={log_id})"
            )
            return log_id

        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            # Don't raise - audit logging should not break the application
            return -1

    def log_login(
        self,
        username: str,
        success: bool,
        user_id: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> int:
        """Log a login attempt."""
        action = AuditAction.LOGIN_SUCCESS if success else AuditAction.LOGIN_FAILED
        return self.log(
            action_type=action,
            resource_type='auth',
            details={'username': username},
            success=success,
            error_message=error_message,
            user_id=user_id,
            username=username
        )

    def log_logout(self, user_id: int, username: str) -> int:
        """Log a logout event."""
        return self.log(
            action_type=AuditAction.LOGOUT,
            resource_type='auth',
            success=True,
            user_id=user_id,
            username=username
        )

    def log_screening(
        self,
        action: str,
        screening_id: str,
        details: Optional[Dict] = None
    ) -> int:
        """Log a screening-related action."""
        return self.log(
            action_type=action,
            resource_type='screening',
            resource_id=screening_id,
            details=details
        )

    def log_user_management(
        self,
        action: str,
        target_user_id: int,
        details: Optional[Dict] = None
    ) -> int:
        """Log a user management action."""
        return self.log(
            action_type=action,
            resource_type='user',
            resource_id=str(target_user_id),
            details=details
        )


def audit_action(
    action_type: str,
    resource_type: Optional[str] = None,
    get_resource_id: Optional[Callable] = None,
    include_request_data: bool = False
):
    """
    Decorator for automatic audit logging of route actions.

    Args:
        action_type: Type of action (from AuditAction)
        resource_type: Type of resource affected
        get_resource_id: Callable to extract resource ID from args/kwargs
        include_request_data: Include request form/json data in details

    Usage:
        @audit_action(AuditAction.SCREENING_CREATE, 'screening')
        def create_screening():
            ...

        @audit_action(
            AuditAction.SCREENING_VIEW,
            'screening',
            get_resource_id=lambda *args, **kwargs: kwargs.get('record_id')
        )
        def view_screening(record_id):
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            resource_id = None
            if get_resource_id:
                try:
                    resource_id = get_resource_id(*args, **kwargs)
                except Exception:
                    pass

            # Collect details
            details = {}
            if include_request_data:
                if request.is_json:
                    details['request_json'] = request.get_json(silent=True)
                elif request.form:
                    # Don't log passwords
                    details['request_form'] = {
                        k: v for k, v in request.form.items()
                        if 'password' not in k.lower()
                    }

            try:
                result = f(*args, **kwargs)

                # Try to extract resource_id from result
                if resource_id is None:
                    if isinstance(result, dict):
                        resource_id = result.get('record_id') or result.get('id')
                    elif hasattr(result, 'get_json'):
                        try:
                            json_data = result.get_json(silent=True)
                            if json_data:
                                resource_id = json_data.get('record_id') or json_data.get('id')
                        except Exception:
                            pass

                # Log success
                if hasattr(g, 'audit_logger'):
                    g.audit_logger.log(
                        action_type=action_type,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        details=details if details else None,
                        success=True
                    )

                return result

            except Exception as e:
                # Log failure
                if hasattr(g, 'audit_logger'):
                    g.audit_logger.log(
                        action_type=action_type,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        details=details if details else None,
                        success=False,
                        error_message=str(e)
                    )
                raise

        return decorated_function
    return decorator


def get_audit_logs(
    conn: sqlite3.Connection,
    user_id: Optional[int] = None,
    action_type: Optional[str] = None,
    resource_type: Optional[str] = None,
    success: Optional[bool] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict]:
    """
    Query audit logs with filters.

    Args:
        conn: Database connection
        user_id: Filter by user ID
        action_type: Filter by action type
        resource_type: Filter by resource type
        success: Filter by success status
        start_date: Filter by start date
        end_date: Filter by end date
        limit: Maximum number of results
        offset: Offset for pagination

    Returns:
        List of audit log entries as dictionaries
    """
    query = """
        SELECT
            al.id, al.timestamp, al.user_id, al.username, al.action_type,
            al.resource_type, al.resource_id, al.ip_address, al.success,
            al.error_message, al.details
        FROM audit_logs al
        WHERE 1=1
    """
    params = []

    if user_id is not None:
        query += " AND al.user_id = ?"
        params.append(user_id)

    if action_type:
        query += " AND al.action_type = ?"
        params.append(action_type)

    if resource_type:
        query += " AND al.resource_type = ?"
        params.append(resource_type)

    if success is not None:
        query += " AND al.success = ?"
        params.append(success)

    if start_date:
        query += " AND al.timestamp >= ?"
        params.append(start_date.isoformat())

    if end_date:
        query += " AND al.timestamp <= ?"
        params.append(end_date.isoformat())

    query += " ORDER BY al.timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = conn.execute(query, params)
    logs = []

    for row in cursor.fetchall():
        log_entry = dict(row)
        # Parse JSON details
        if log_entry.get('details'):
            try:
                log_entry['details'] = json.loads(log_entry['details'])
            except json.JSONDecodeError:
                pass
        logs.append(log_entry)

    return logs


def get_audit_log_count(
    conn: sqlite3.Connection,
    user_id: Optional[int] = None,
    action_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> int:
    """Get total count of audit logs matching filters."""
    query = "SELECT COUNT(*) FROM audit_logs WHERE 1=1"
    params = []

    if user_id is not None:
        query += " AND user_id = ?"
        params.append(user_id)

    if action_type:
        query += " AND action_type = ?"
        params.append(action_type)

    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date.isoformat())

    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date.isoformat())

    cursor = conn.execute(query, params)
    return cursor.fetchone()[0]


def get_audit_summary(
    conn: sqlite3.Connection,
    days: int = 7
) -> Dict[str, Any]:
    """
    Get audit log summary statistics.

    Args:
        conn: Database connection
        days: Number of days to include

    Returns:
        Dictionary with summary statistics
    """
    start_date = datetime.utcnow() - timedelta(days=days)

    # Total events
    total = get_audit_log_count(conn, start_date=start_date)

    # Events by action type
    cursor = conn.execute("""
        SELECT action_type, COUNT(*) as count
        FROM audit_logs
        WHERE timestamp >= ?
        GROUP BY action_type
        ORDER BY count DESC
    """, (start_date.isoformat(),))
    by_action = {row['action_type']: row['count'] for row in cursor.fetchall()}

    # Failed events
    cursor = conn.execute("""
        SELECT COUNT(*) FROM audit_logs
        WHERE timestamp >= ? AND success = 0
    """, (start_date.isoformat(),))
    failed = cursor.fetchone()[0]

    # Active users
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM audit_logs
        WHERE timestamp >= ? AND user_id IS NOT NULL
    """, (start_date.isoformat(),))
    active_users = cursor.fetchone()[0]

    # Recent failed logins
    cursor = conn.execute("""
        SELECT username, COUNT(*) as count, MAX(timestamp) as last_attempt
        FROM audit_logs
        WHERE action_type = 'login_failed'
          AND timestamp >= ?
        GROUP BY username
        ORDER BY count DESC
        LIMIT 10
    """, (start_date.isoformat(),))
    failed_logins = [dict(row) for row in cursor.fetchall()]

    return {
        'period_days': days,
        'total_events': total,
        'failed_events': failed,
        'active_users': active_users,
        'events_by_action': by_action,
        'failed_logins': failed_logins
    }


def cleanup_old_audit_logs(
    conn: sqlite3.Connection,
    retention_days: int = 365
) -> int:
    """
    Delete audit logs older than retention period.

    Args:
        conn: Database connection
        retention_days: Number of days to retain logs

    Returns:
        Number of logs deleted
    """
    cutoff = datetime.utcnow() - timedelta(days=retention_days)

    cursor = conn.execute("""
        DELETE FROM audit_logs WHERE timestamp < ?
    """, (cutoff.isoformat(),))
    conn.commit()

    count = cursor.rowcount
    if count > 0:
        logger.info(f"Cleaned up {count} audit logs older than {retention_days} days")
    return count
