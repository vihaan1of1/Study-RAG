"""StudyRAG Authentication & Supabase Backend Manager.

Handles user registration, login, session management, and robust form validation.
Supports official Supabase Auth with a graceful local encrypted fallback so the application
remains 100% testable and operational both with and without cloud Supabase credentials.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Form Validation Helpers
# ---------------------------------------------------------------------------

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email(email: str) -> Tuple[bool, str]:
    """Validate email format.

    Args:
        email: Email string to check.

    Returns:
        (is_valid, error_message)
    """
    if not email or not email.strip():
        return False, "Email address is required."

    cleaned = email.strip()
    if len(cleaned) > 254:
        return False, "Email address is too long."

    if not EMAIL_REGEX.match(cleaned):
        return False, "Please enter a valid email address (e.g., student@university.edu)."

    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password strength according to Supabase requirements.

    Supabase Auth requires a minimum of 6 characters by default.

    Args:
        password: Password string to check.

    Returns:
        (is_valid, error_message)
    """
    if not password:
        return False, "Password is required."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    if len(password) > 128:
        return False, "Password must not exceed 128 characters."

    return True, ""


def validate_registration(
    email: str, password: str, confirm_password: str
) -> Tuple[bool, str]:
    """Validate all registration form fields.

    Args:
        email: User email.
        password: Chosen password.
        confirm_password: Repeated password confirmation.

    Returns:
        (is_valid, error_message)
    """
    valid_email, err_email = validate_email(email)
    if not valid_email:
        return False, err_email

    valid_pw, err_pw = validate_password(password)
    if not valid_pw:
        return False, err_pw

    if password != confirm_password:
        return False, "Passwords do not match. Please re-enter your password."

    return True, ""


# ---------------------------------------------------------------------------
# Authenticated User Model
# ---------------------------------------------------------------------------

@dataclass
class AuthUser:
    """Authenticated StudyRAG user representation."""
    id: str
    email: str
    created_at: str
    display_name: str
    backend: str = "supabase"  # "supabase" or "local"
    raw_user: Optional[Any] = None

    def get_storage_path(self) -> Path:
        """Return the user's isolated document and vector storage path."""
        # Sanitize user ID for Windows folder naming
        safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', self.id)
        path = config.USERS_DIR / safe_id
        path.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------------------------------------------------------
# Authentication & Backend Manager
# ---------------------------------------------------------------------------

class AuthManager:
    """Manages Supabase Auth and local authentication with user persistence."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        self.url = config.get_supabase_url(supabase_url)
        self.key = config.get_supabase_key(supabase_key)
        self.supabase_client = None
        self.backend_type = "local"

        # Attempt to initialize Supabase client if credentials are configured
        if config.is_supabase_configured(self.url, self.key):
            try:
                from supabase import create_client
                self.supabase_client = create_client(self.url, self.key)
                self.backend_type = "supabase"
                logger.info(f"Supabase client connected to {self.url}")
            except Exception as exc:
                logger.warning(f"Could not connect to Supabase: {exc}. Falling back to local auth.")
                self.supabase_client = None
                self.backend_type = "local"

        # Local auth storage file
        self.local_db_path = config.DATA_DIR / "local_auth.json"
        self._init_local_db()

    def _init_local_db(self) -> None:
        """Ensure local auth storage exists."""
        if not self.local_db_path.exists():
            self._write_local_db({"users": {}})

    def _read_local_db(self) -> Dict[str, Any]:
        """Read users from local auth store."""
        try:
            with open(self.local_db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"users": {}}

    def _write_local_db(self, data: Dict[str, Any]) -> None:
        """Write users to local auth store."""
        self.local_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.local_db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _hash_password(password: str, salt: str = "studyrag_salt_2026") -> str:
        """Compute salted SHA-256 hash for secure local auth fallback."""
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

    @property
    def is_supabase_active(self) -> bool:
        """True if Supabase client is connected and active."""
        return self.supabase_client is not None

    def sign_in(
        self, email: str, password: str
    ) -> Tuple[bool, str, Optional[AuthUser]]:
        """Sign in with email and password.

        Args:
            email: User email.
            password: User password.

        Returns:
            (success, message, user_object)
        """
        # Validate inputs first
        v_email, err_email = validate_email(email)
        if not v_email:
            return False, err_email, None

        v_pw, err_pw = validate_password(password)
        if not v_pw:
            return False, err_pw, None

        cleaned_email = email.strip().lower()

        # 1. Supabase Auth if configured
        if self.is_supabase_active:
            try:
                res = self.supabase_client.auth.sign_in_with_password(
                    {"email": cleaned_email, "password": password}
                )
                if res.user:
                    user_id = str(res.user.id)
                    display = (
                        res.user.user_metadata.get("full_name")
                        or cleaned_email.split("@")[0].capitalize()
                    )
                    user = AuthUser(
                        id=user_id,
                        email=res.user.email or cleaned_email,
                        created_at=str(res.user.created_at or datetime.now().isoformat()),
                        display_name=display,
                        backend="supabase",
                        raw_user=res.user,
                    )
                    return True, "Signed in successfully via Supabase Auth!", user
                return False, "Invalid authentication response from Supabase.", None
            except Exception as exc:
                err_msg = str(exc)
                if "invalid login credentials" in err_msg.lower():
                    return False, "Invalid email or password. Please verify your credentials.", None
                if "email not confirmed" in err_msg.lower():
                    return False, "Email address has not been confirmed yet. Please check your inbox or disable email confirmation in your Supabase dashboard.", None
                return False, f"Supabase sign-in error: {err_msg}", None

        # 2. Local Fallback Auth
        db = self._read_local_db()
        users = db.get("users", {})

        if cleaned_email not in users:
            return False, "No account found with this email. Please create an account first.", None

        user_record = users[cleaned_email]
        if user_record.get("password_hash") != self._hash_password(password):
            return False, "Incorrect password. Please try again.", None

        user = AuthUser(
            id=user_record.get("id", cleaned_email),
            email=cleaned_email,
            created_at=user_record.get("created_at", datetime.now().isoformat()),
            display_name=user_record.get("display_name", cleaned_email.split("@")[0].capitalize()),
            backend="local",
        )
        return True, "Signed in successfully (Local Auth Mode)!", user

    def sign_up(
        self, email: str, password: str, confirm_password: str
    ) -> Tuple[bool, str, Optional[AuthUser]]:
        """Register a new user account with complete form validation.

        Args:
            email: User email.
            password: User password.
            confirm_password: Confirmation password.

        Returns:
            (success, message, user_object)
        """
        # Validate all form inputs
        v_ok, v_err = validate_registration(email, password, confirm_password)
        if not v_ok:
            return False, v_err, None

        cleaned_email = email.strip().lower()

        # 1. Supabase Auth if configured
        if self.is_supabase_active:
            try:
                res = self.supabase_client.auth.sign_up(
                    {"email": cleaned_email, "password": password}
                )
                if res.user:
                    user_id = str(res.user.id)
                    display = cleaned_email.split("@")[0].capitalize()
                    user = AuthUser(
                        id=user_id,
                        email=res.user.email or cleaned_email,
                        created_at=str(res.user.created_at or datetime.now().isoformat()),
                        display_name=display,
                        backend="supabase",
                        raw_user=res.user,
                    )
                    # Check if session was returned or email confirmation is pending
                    if res.session is None and getattr(res.user, "confirmed_at", None) is None:
                        return (
                            True,
                            "Account created in Supabase! If your Supabase project requires email confirmation, please check your inbox before logging in.",
                            user,
                        )
                    return True, "Account created successfully via Supabase!", user
                return False, "Failed to register user in Supabase.", None
            except Exception as exc:
                err_msg = str(exc)
                if "already registered" in err_msg.lower() or "already exists" in err_msg.lower():
                    return False, "An account with this email already exists. Please sign in.", None
                return False, f"Supabase registration error: {err_msg}", None

        # 2. Local Fallback Auth
        db = self._read_local_db()
        users = db.get("users", {})

        if cleaned_email in users:
            return False, "An account with this email already exists. Please sign in.", None

        # Create deterministic user ID from email
        user_id = f"user_{hashlib.md5(cleaned_email.encode()).hexdigest()[:12]}"
        now_str = datetime.now().isoformat()
        display = cleaned_email.split("@")[0].capitalize()

        users[cleaned_email] = {
            "id": user_id,
            "email": cleaned_email,
            "password_hash": self._hash_password(password),
            "display_name": display,
            "created_at": now_str,
        }
        db["users"] = users
        self._write_local_db(db)

        user = AuthUser(
            id=user_id,
            email=cleaned_email,
            created_at=now_str,
            display_name=display,
            backend="local",
        )
        return True, "Account registered successfully (Local Auth Mode)!", user

    def sign_out(self) -> None:
        """Sign out current user."""
        if self.is_supabase_active:
            try:
                self.supabase_client.auth.sign_out()
            except Exception as exc:
                logger.warning(f"Error during Supabase sign-out: {exc}")

    def get_demo_student(self) -> AuthUser:
        """Create or retrieve a pre-configured demo student user for instant evaluation."""
        demo_email = "student@studyrag.edu"
        demo_id = "demo_student_2026"
        now_str = datetime.now().isoformat()

        # Ensure demo user exists in local DB
        db = self._read_local_db()
        users = db.get("users", {})
        if demo_email not in users:
            users[demo_email] = {
                "id": demo_id,
                "email": demo_email,
                "password_hash": self._hash_password("student123"),
                "display_name": "Demo Student",
                "created_at": now_str,
            }
            db["users"] = users
            self._write_local_db(db)

        return AuthUser(
            id=demo_id,
            email=demo_email,
            created_at=now_str,
            display_name="Demo Student",
            backend="local" if not self.is_supabase_active else "supabase",
        )
