-- ASD Prediction System Database Schema
-- Authentication, Authorization, and Accountability (AAA)

-- =====================================================
-- ROLES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Default roles
INSERT OR IGNORE INTO roles (name, display_name, description) VALUES
    ('admin', 'System Administrator', 'Full system access including user management and audit logs'),
    ('pediatrician', 'Pediatrician', 'Full screening access, can view all records and export data'),
    ('nurse', 'Nurse', 'Can create and view own screenings'),
    ('chw', 'Community Health Worker', 'Basic screening capabilities, view own records only');

-- =====================================================
-- USERS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    role_id INTEGER NOT NULL,
    facility VARCHAR(200),
    is_active BOOLEAN DEFAULT 1,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    password_changed_at DATETIME,
    must_change_password BOOLEAN DEFAULT 0,
    created_by INTEGER,
    FOREIGN KEY (role_id) REFERENCES roles(id),
    FOREIGN KEY (created_by) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role_id);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- =====================================================
-- PERMISSIONS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS permissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(150) NOT NULL,
    description TEXT,
    resource_type VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Default permissions
INSERT OR IGNORE INTO permissions (name, display_name, resource_type, action) VALUES
    ('screening.create', 'Create Screening', 'screening', 'create'),
    ('screening.view', 'View Own Screenings', 'screening', 'view'),
    ('screening.view_all', 'View All Screenings', 'screening', 'view_all'),
    ('screening.export', 'Export Screenings', 'screening', 'export'),
    ('user.create', 'Create Users', 'user', 'create'),
    ('user.edit', 'Edit Users', 'user', 'edit'),
    ('user.view', 'View Users', 'user', 'view'),
    ('user.deactivate', 'Deactivate Users', 'user', 'deactivate'),
    ('audit.view', 'View Audit Logs', 'audit', 'view'),
    ('report.generate', 'Generate Reports', 'report', 'generate'),
    ('report.view', 'View Reports', 'report', 'view'),
    ('settings.manage', 'Manage System Settings', 'settings', 'manage');

-- =====================================================
-- ROLE-PERMISSION MAPPING TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS role_permissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_id INTEGER NOT NULL,
    permission_id INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (role_id) REFERENCES roles(id),
    FOREIGN KEY (permission_id) REFERENCES permissions(id),
    UNIQUE(role_id, permission_id)
);

-- Admin permissions (all)
INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id FROM roles r, permissions p WHERE r.name = 'admin';

-- Pediatrician permissions
INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id FROM roles r, permissions p
WHERE r.name = 'pediatrician' AND p.name IN (
    'screening.create', 'screening.view', 'screening.view_all',
    'screening.export', 'report.generate', 'report.view'
);

-- Nurse permissions
INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id FROM roles r, permissions p
WHERE r.name = 'nurse' AND p.name IN (
    'screening.create', 'screening.view', 'report.view'
);

-- CHW permissions
INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id FROM roles r, permissions p
WHERE r.name = 'chw' AND p.name IN (
    'screening.create', 'screening.view'
);

-- =====================================================
-- SESSIONS TABLE (for tracking active sessions)
-- =====================================================
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

-- =====================================================
-- AUDIT LOGS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER,
    username VARCHAR(50),
    session_id VARCHAR(255),
    action_type VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    ip_address VARCHAR(45),
    user_agent TEXT,
    request_method VARCHAR(10),
    request_path TEXT,
    details TEXT,
    success BOOLEAN DEFAULT 1,
    error_message TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action_type);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_success ON audit_logs(success);

-- =====================================================
-- PASSWORD HISTORY TABLE (optional - for preventing reuse)
-- =====================================================
CREATE TABLE IF NOT EXISTS password_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_password_history_user ON password_history(user_id);

-- =====================================================
-- PASSWORD RESET REQUESTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS password_reset_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    username VARCHAR(50) NOT NULL,
    reason TEXT,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, approved, rejected
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed_at DATETIME,
    processed_by INTEGER,
    admin_notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (processed_by) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_reset_requests_user ON password_reset_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_reset_requests_status ON password_reset_requests(status);
CREATE INDEX IF NOT EXISTS idx_reset_requests_created ON password_reset_requests(created_at);
