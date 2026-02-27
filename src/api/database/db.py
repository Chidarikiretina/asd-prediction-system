"""
Database Connection Management for ASD Prediction System.
Handles SQLite connection, initialization, and cleanup.
"""

import sqlite3
import logging
import os
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database file path
_DB_PATH: Optional[Path] = None
_connection_pool = {}


def get_db_path() -> Path:
    """Get the database file path."""
    global _DB_PATH
    if _DB_PATH is None:
        env_db_path = os.getenv('ASD_DB_PATH')
        if env_db_path:
            _DB_PATH = Path(env_db_path)
            return _DB_PATH
        # Default path: project_root/data/asd_system.db
        project_root = Path(__file__).parent.parent.parent.parent
        _DB_PATH = project_root / 'data' / 'asd_system.db'
    return _DB_PATH


def set_db_path(path: Path) -> None:
    """Set custom database file path."""
    global _DB_PATH
    _DB_PATH = path


def get_db_connection() -> sqlite3.Connection:
    """
    Get a database connection.
    Creates the database and schema if it doesn't exist.
    """
    db_path = get_db_path()

    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if database needs initialization
    needs_init = not db_path.exists()

    # Create connection with row factory for dict-like access
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")

    # Initialize schema if new database
    if needs_init:
        logger.info(f"Initializing new database at {db_path}")
        init_database(conn)

    return conn


def init_database(conn: Optional[sqlite3.Connection] = None) -> None:
    """
    Initialize database with schema.
    Creates all tables and populates default data.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        # Read and execute schema file
        schema_path = Path(__file__).parent / 'schema.sql'

        if schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            conn.executescript(schema_sql)
            conn.commit()
            logger.info("Database schema initialized successfully")
        else:
            logger.warning(f"Schema file not found at {schema_path}")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        if close_conn:
            conn.close()


def close_database(conn: sqlite3.Connection) -> None:
    """Close database connection."""
    if conn:
        try:
            conn.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")


@contextmanager
def db_transaction(conn: sqlite3.Connection):
    """
    Context manager for database transactions.
    Automatically commits on success, rolls back on error.
    """
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction rolled back due to error: {e}")
        raise


def execute_query(conn: sqlite3.Connection, query: str, params: tuple = ()) -> sqlite3.Cursor:
    """Execute a query and return cursor."""
    cursor = conn.cursor()
    cursor.execute(query, params)
    return cursor


def fetch_one(conn: sqlite3.Connection, query: str, params: tuple = ()) -> Optional[dict]:
    """Execute query and fetch one result as dict."""
    cursor = execute_query(conn, query, params)
    row = cursor.fetchone()
    if row:
        return dict(row)
    return None


def fetch_all(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list:
    """Execute query and fetch all results as list of dicts."""
    cursor = execute_query(conn, query, params)
    return [dict(row) for row in cursor.fetchall()]


def check_database_health() -> dict:
    """
    Check database health and return status.
    """
    try:
        conn = get_db_connection()

        # Check tables exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [row[0] for row in cursor.fetchall()]

        # Check user count
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

        # Check audit log count
        audit_count = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]

        conn.close()

        return {
            'status': 'healthy',
            'tables': tables,
            'user_count': user_count,
            'audit_log_count': audit_count,
            'db_path': str(get_db_path())
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }
