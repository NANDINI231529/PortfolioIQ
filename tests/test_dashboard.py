import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DashboardEntryTests(unittest.TestCase):
    def test_unauthenticated_direct_dashboard_state_shows_landing_page(self):
        app_file = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_file)).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        app.session_state["page"] = "dashboard"
        app.run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        button_labels = [button.label for button in app.button]
        self.assertIn("Log in", button_labels)
        self.assertIn("Create account", button_labels)
        self.assertNotIn("Log out", button_labels)


if __name__ == "__main__":
    unittest.main()
