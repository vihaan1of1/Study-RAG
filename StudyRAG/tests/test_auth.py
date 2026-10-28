"""Tests for StudyRAG Authentication & Form Validation.

Validates email formatting, password length rules, password confirmation match,
and local authentication state persistence.
"""

import pytest
from src.auth import (
    validate_email,
    validate_password,
    validate_registration,
    AuthManager,
)


def test_validate_email():
    """Verify email regex and format validation."""
    # Valid emails
    valid_emails = [
        "student@university.edu",
        "john.doe@gmail.com",
        "alex_123@sub.domain.org",
    ]
    for e in valid_emails:
        ok, msg = validate_email(e)
        assert ok, f"Expected '{e}' to be valid, got error: {msg}"

    # Invalid emails
    invalid_emails = [
        "",
        "   ",
        "plainaddress",
        "@missingusername.com",
        "username@.com",
        "username@domain",
    ]
    for e in invalid_emails:
        ok, msg = validate_email(e)
        assert not ok, f"Expected '{e}' to be invalid"
        assert len(msg) > 0


def test_validate_password():
    """Verify password length rules."""
    # Too short
    ok, msg = validate_password("12345")
    assert not ok
    assert "at least 6 characters" in msg

    # Empty
    ok, msg = validate_password("")
    assert not ok
    assert "required" in msg

    # Valid
    ok, msg = validate_password("secret123")
    assert ok
    assert msg == ""


def test_validate_registration():
    """Verify registration form composite validation."""
    # Mismatched passwords
    ok, msg = validate_registration("student@edu.com", "pass123", "pass456")
    assert not ok
    assert "Passwords do not match" in msg

    # Invalid email with matching passwords
    ok, msg = validate_registration("invalid-email", "pass123", "pass123")
    assert not ok
    assert "valid email" in msg

    # Short password with matching confirmation
    ok, msg = validate_registration("student@edu.com", "123", "123")
    assert not ok
    assert "at least 6 characters" in msg

    # Valid registration
    ok, msg = validate_registration("student@edu.com", "strongpass123", "strongpass123")
    assert ok
    assert msg == ""


def test_local_auth_flow(tmp_path, monkeypatch):
    """Verify complete registration, login, and rejection of invalid credentials."""
    from src import auth
    # Redirect local auth db to temporary path
    test_db = tmp_path / "test_auth.json"
    monkeypatch.setattr(auth.config, "DATA_DIR", tmp_path)

    mgr = AuthManager()
    mgr.local_db_path = test_db
    mgr._init_local_db()

    # 1. Register new user
    ok, msg, user = mgr.sign_up("alice@college.edu", "alicePassword123", "alicePassword123")
    assert ok, f"Registration failed: {msg}"
    assert user is not None
    assert user.email == "alice@college.edu"

    # 2. Prevent duplicate registration
    ok_dup, msg_dup, _ = mgr.sign_up("alice@college.edu", "alicePassword123", "alicePassword123")
    assert not ok_dup
    assert "already exists" in msg_dup

    # 3. Wrong password rejection
    ok_wrong, msg_wrong, _ = mgr.sign_in("alice@college.edu", "wrongPassword")
    assert not ok_wrong
    assert "Incorrect password" in msg_wrong

    # 4. Correct login
    ok_login, msg_login, logged_user = mgr.sign_in("alice@college.edu", "alicePassword123")
    assert ok_login
    assert logged_user is not None
    assert logged_user.email == "alice@college.edu"
