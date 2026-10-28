"""Small SQLite data layer. Passwords are never handled in this module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from typing import Optional


@dataclass(frozen=True)
class UserRecord:
    user_id: int
    full_name: str
    email: str
    password_hash: str
    password_salt: str
    created_at: str


class Database:
    """SQLite account store using short-lived connections and parameterized SQL."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email_normalized TEXT NOT NULL UNIQUE,
                    email_display TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    password_algorithm TEXT NOT NULL DEFAULT 'pbkdf2_sha256_v1',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def create_user(
        self, full_name: str, email: str, password_hash: str, password_salt: str
    ) -> int:
        clean_name = full_name.strip()
        normalized_email = email.strip().casefold()
        if not clean_name or not normalized_email or not password_hash or not password_salt:
            raise ValueError("Required user fields are missing.")
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    "INSERT INTO users (full_name, email_normalized, email_display, password_hash, password_salt) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (clean_name, normalized_email, email.strip(), password_hash, password_salt),
                )
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            if "email_normalized" in str(exc).lower() or "unique" in str(exc).lower():
                raise ValueError("An account with this email address already exists.") from exc
            raise

    def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        normalized_email = email.strip().casefold()
        if not normalized_email:
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, full_name, email_display, password_hash, password_salt, created_at "
                "FROM users WHERE email_normalized = ?",
                (normalized_email,),
            ).fetchone()
        return self._record(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[UserRecord]:
        if not isinstance(user_id, int) or user_id <= 0:
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, full_name, email_display, password_hash, password_salt, created_at "
                "FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return self._record(row) if row else None

    @staticmethod
    def _record(row: sqlite3.Row) -> UserRecord:
        return UserRecord(
            user_id=int(row["user_id"]), full_name=row["full_name"],
            email=row["email_display"], password_hash=row["password_hash"],
            password_salt=row["password_salt"], created_at=row["created_at"],
        )
