"""Educational session authentication with salted PBKDF2-HMAC-SHA256."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from typing import MutableMapping

from database.db import Database

PBKDF2_ITERATIONS = 600_000
PASSWORD_MIN_LENGTH = 8
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class RegistrationError(ValueError):
    """A safe, user-facing registration validation error."""


class AuthenticationError(ValueError):
    """Generic invalid-credentials error."""


class AuthService:
    def __init__(self, database: Database, iterations: int = PBKDF2_ITERATIONS):
        self.database = database
        self.iterations = int(iterations)

    @staticmethod
    def validate_registration(full_name: str, email: str, password: str, confirmation: str) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (full_name, email, password, confirmation)):
            raise RegistrationError("Please complete every field.")
        if not EMAIL_RE.fullmatch(email.strip()):
            raise RegistrationError("Enter a valid email address.")
        if len(password) < PASSWORD_MIN_LENGTH:
            raise RegistrationError(f"Use a password with at least {PASSWORD_MIN_LENGTH} characters.")
        if not hmac.compare_digest(password, confirmation):
            raise RegistrationError("The passwords do not match.")

    @staticmethod
    def hash_password(password: str, salt: bytes | None = None, iterations: int = PBKDF2_ITERATIONS) -> tuple[str, str]:
        salt_bytes = salt or secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, iterations)
        return digest.hex(), salt_bytes.hex()

    @staticmethod
    def verify_password(password: str, password_hash: str, password_salt: str, iterations: int = PBKDF2_ITERATIONS) -> bool:
        try:
            salt = bytes.fromhex(password_salt)
            expected = bytes.fromhex(password_hash)
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(actual, expected)

    def register(self, full_name: str, email: str, password: str, confirmation: str) -> int:
        self.validate_registration(full_name, email, password, confirmation)
        digest, salt = self.hash_password(password, iterations=self.iterations)
        try:
            return self.database.create_user(full_name, email, digest, salt)
        except ValueError as exc:
            raise RegistrationError(str(exc)) from exc

    def login(self, email: str, password: str) -> dict[str, int | str]:
        user = self.database.get_user_by_email(email if isinstance(email, str) else "")
        if not isinstance(password, str):
            password = ""
        if user is None:
            # Match the expensive work done for an existing user to reduce account-enumeration timing clues.
            hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes(16), self.iterations)
            raise AuthenticationError("Invalid email or password.")
        if not self.verify_password(password, user.password_hash, user.password_salt, self.iterations):
            raise AuthenticationError("Invalid email or password.")
        return {"user_id": user.user_id, "full_name": user.full_name}


def set_auth_session(session: MutableMapping[str, object], user: dict[str, int | str]) -> None:
    """Persist only the minimum display identity; never password material or email."""
    session["authenticated"] = True
    session["user_id"] = int(user["user_id"])
    session["full_name"] = str(user["full_name"])


def clear_auth_session(session: MutableMapping[str, object]) -> None:
    for key in ("authenticated", "user_id", "full_name"):
        session.pop(key, None)


def is_authenticated(session: MutableMapping[str, object]) -> bool:
    return session.get("authenticated") is True and isinstance(session.get("user_id"), int)
