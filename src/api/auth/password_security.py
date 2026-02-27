"""
Password Security Module for ASD Prediction System.
Provides bcrypt hashing, password validation, and account lockout.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

# Try to import bcrypt, fall back to hashlib if not available
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    import hashlib
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not available, using SHA-256 fallback (less secure)")

# Configuration constants
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
BCRYPT_ROUNDS = 12
LOCKOUT_THRESHOLD = 3
LOCKOUT_DURATION_MINUTES = 15


class PasswordPolicy:
    """
    Password complexity requirements and validation.
    Enforces strong password policies.
    """

    def __init__(
        self,
        min_length: int = PASSWORD_MIN_LENGTH,
        max_length: int = PASSWORD_MAX_LENGTH,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_digit: bool = True,
        require_special: bool = True
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digit = require_digit
        self.require_special = require_special
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"

    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against complexity requirements.

        Args:
            password: The password to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        if not password:
            return (False, ["Password is required"])

        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters")

        if len(password) > self.max_length:
            errors.append(f"Password must be no more than {self.max_length} characters")

        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if self.require_digit and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")

        if self.require_special:
            if not any(c in self.special_chars for c in password):
                errors.append("Password must contain at least one special character (!@#$%^&*()_+-=[]{}|;:',.<>?/~`)")

        return (len(errors) == 0, errors)

    def get_requirements_text(self) -> str:
        """Get human-readable password requirements."""
        reqs = [f"At least {self.min_length} characters"]
        if self.require_uppercase:
            reqs.append("One uppercase letter")
        if self.require_lowercase:
            reqs.append("One lowercase letter")
        if self.require_digit:
            reqs.append("One digit")
        if self.require_special:
            reqs.append("One special character")
        return "; ".join(reqs)


class PasswordHasher:
    """
    Secure password hashing using bcrypt.
    Falls back to SHA-256 if bcrypt is not available.
    """

    def __init__(self, rounds: int = BCRYPT_ROUNDS):
        self.rounds = rounds

    def hash_password(self, password: str) -> str:
        """
        Generate secure hash of password.

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        if BCRYPT_AVAILABLE:
            password_bytes = password.encode('utf-8')
            salt = bcrypt.gensalt(rounds=self.rounds)
            hashed = bcrypt.hashpw(password_bytes, salt)
            return hashed.decode('utf-8')
        else:
            # Fallback to SHA-256 (less secure, but functional)
            import hashlib
            return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify password against stored hash.

        Args:
            password: Plain text password to verify
            password_hash: Stored password hash

        Returns:
            True if password matches, False otherwise
        """
        try:
            if BCRYPT_AVAILABLE:
                password_bytes = password.encode('utf-8')
                hash_bytes = password_hash.encode('utf-8')
                return bcrypt.checkpw(password_bytes, hash_bytes)
            else:
                # Fallback verification for SHA-256
                import hashlib
                return hashlib.sha256(password.encode()).hexdigest() == password_hash
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        """
        Check if password hash needs to be upgraded.
        Returns True for old SHA-256 hashes when bcrypt is available.
        """
        if BCRYPT_AVAILABLE:
            # bcrypt hashes start with $2b$ or $2a$
            return not password_hash.startswith('$2')
        return False


class AccountLockout:
    """
    Account lockout management after failed login attempts.
    Prevents brute force attacks.
    """

    def __init__(
        self,
        threshold: int = LOCKOUT_THRESHOLD,
        duration_minutes: int = LOCKOUT_DURATION_MINUTES
    ):
        self.threshold = threshold
        self.duration = timedelta(minutes=duration_minutes)

    def check_lockout(
        self,
        failed_attempts: int,
        locked_until: Optional[datetime]
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if account is currently locked.

        Args:
            failed_attempts: Number of consecutive failed login attempts
            locked_until: Datetime when lockout expires (or None)

        Returns:
            Tuple of (is_locked, minutes_remaining)
        """
        if locked_until:
            now = datetime.utcnow()
            if locked_until > now:
                remaining = (locked_until - now).total_seconds() / 60
                return (True, int(remaining) + 1)
        return (False, None)

    def should_lock(self, failed_attempts: int) -> bool:
        """
        Determine if account should be locked based on failed attempts.

        Args:
            failed_attempts: Number of consecutive failed login attempts

        Returns:
            True if account should be locked
        """
        return failed_attempts >= self.threshold

    def get_lockout_until(self) -> datetime:
        """
        Get datetime when lockout should expire.

        Returns:
            Datetime when account can be unlocked
        """
        return datetime.utcnow() + self.duration

    def get_attempts_remaining(self, failed_attempts: int) -> int:
        """
        Get number of attempts remaining before lockout.

        Args:
            failed_attempts: Current number of failed attempts

        Returns:
            Number of attempts remaining (minimum 0)
        """
        remaining = self.threshold - failed_attempts
        return max(0, remaining)


def generate_temporary_password(length: int = 12) -> str:
    """
    Generate a temporary password that meets complexity requirements.

    Args:
        length: Length of password to generate (minimum 12)

    Returns:
        Random password string
    """
    import secrets
    import string

    length = max(12, length)

    # Ensure at least one of each required character type
    password = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%^&*()_+-=")
    ]

    # Fill remaining length with random characters
    all_chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
    password.extend(secrets.choice(all_chars) for _ in range(length - 4))

    # Shuffle to avoid predictable pattern
    secrets.SystemRandom().shuffle(password)

    return ''.join(password)
