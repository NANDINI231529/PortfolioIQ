import tempfile
import unittest
from pathlib import Path

from authentication.auth import AuthService, AuthenticationError, RegistrationError, clear_auth_session, is_authenticated, set_auth_session
from database.db import Database


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        project_data = Path(__file__).resolve().parents[1] / "data"
        self.temp = tempfile.TemporaryDirectory(dir=project_data)
        self.db = Database(Path(self.temp.name) / "users.sqlite3")
        self.db.initialize()
        self.auth = AuthService(self.db, iterations=10_000)

    def tearDown(self):
        self.temp.cleanup()

    def test_registration_login_and_no_plaintext_storage(self):
        user_id = self.auth.register("Grace Hopper", "Grace@Example.com", "analytical-123", "analytical-123")
        stored = self.db.get_user_by_email("grace@example.com")
        self.assertEqual(stored.user_id, user_id)
        self.assertNotEqual(stored.password_hash, "analytical-123")
        self.assertNotEqual(stored.password_salt, "analytical-123")
        user = self.auth.login("GRACE@example.com", "analytical-123")
        self.assertEqual(user, {"user_id": user_id, "full_name": "Grace Hopper"})
        self.assertNotIn("password", user)
        self.assertNotIn("password_hash", user)

    def test_duplicate_registration(self):
        self.auth.register("One", "same@example.com", "password1", "password1")
        with self.assertRaisesRegex(RegistrationError, "already exists"):
            self.auth.register("Two", "SAME@example.com", "password2", "password2")

    def test_registration_validation(self):
        cases = [
            ("Name", "bad-email", "password1", "password1"),
            (" ", "name@example.com", "password1", "password1"),
            ("Name", "name@example.com", "short", "short"),
            ("Name", "name@example.com", "password1", "different"),
            ("Name", "name@example.com", "", ""),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(RegistrationError):
                self.auth.register(*args)

    def test_hashing_and_password_verification(self):
        digest, salt = self.auth.hash_password("strong-password", iterations=10_000)
        self.assertTrue(self.auth.verify_password("strong-password", digest, salt, iterations=10_000))
        self.assertFalse(self.auth.verify_password("wrong-password", digest, salt, iterations=10_000))
        second_hash, second_salt = self.auth.hash_password("strong-password", iterations=10_000)
        self.assertNotEqual(salt, second_salt)
        self.assertNotEqual(digest, second_hash)

    def test_invalid_credentials_are_rejected_generically(self):
        with self.assertRaises(AuthenticationError):
            self.auth.login("nobody@example.com", "guess")
        self.auth.register("Test User", "person@example.com", "correct-horse", "correct-horse")
        with self.assertRaises(AuthenticationError):
            self.auth.login("person@example.com", "wrong-horse")

    def test_session_login_logout_and_protected_dashboard_guard(self):
        state = {}
        self.assertFalse(is_authenticated(state), "An unauthenticated session must not access the dashboard.")
        user = self.auth.login(*self._registered_credentials())
        set_auth_session(state, user)
        self.assertTrue(is_authenticated(state))
        self.assertNotIn("password", state)
        self.assertNotIn("password_hash", state)
        self.assertNotIn("email", state)
        clear_auth_session(state)
        self.assertFalse(is_authenticated(state), "Logout must clear authenticated dashboard access.")
        self.assertEqual(state, {})

    def _registered_credentials(self):
        self.auth.register("Session User", "session@example.com", "session-password", "session-password")
        return "session@example.com", "session-password"


if __name__ == "__main__":
    unittest.main()
