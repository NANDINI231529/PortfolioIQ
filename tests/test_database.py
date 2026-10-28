import tempfile
import unittest
from pathlib import Path

from database.db import Database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        project_data = Path(__file__).resolve().parents[1] / "data"
        self.temp = tempfile.TemporaryDirectory(dir=project_data)
        self.db = Database(Path(self.temp.name) / "users.sqlite3")
        self.db.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def test_initialization_and_user_creation(self):
        user_id = self.db.create_user("Ada Lovelace", "ADA@example.com", "hash-value", "salt-value")
        user = self.db.get_user_by_id(user_id)
        self.assertIsNotNone(user)
        self.assertEqual(user.full_name, "Ada Lovelace")
        self.assertEqual(user.email, "ADA@example.com")
        self.assertEqual(self.db.get_user_by_email("ada@EXAMPLE.com").user_id, user_id)

    def test_unique_normalized_email(self):
        self.db.create_user("One User", "person@example.com", "hash", "salt")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.db.create_user("Two User", "PERSON@example.com", "hash2", "salt2")

    def test_invalid_fields_and_identifiers(self):
        for values in [(" ", "a@b.com", "h", "s"), ("Name", " ", "h", "s")]:
            with self.assertRaises(ValueError):
                self.db.create_user(*values)
        self.assertIsNone(self.db.get_user_by_id(-1))
        self.assertIsNone(self.db.get_user_by_email(" "))


if __name__ == "__main__":
    unittest.main()
