import unittest

import pandas as pd

from components.ui import csv_bytes


class CsvExportTests(unittest.TestCase):
    def test_csv_bytes_preserve_computed_columns_and_values(self):
        computed = pd.DataFrame(
            {"Expected Annualized Return": [0.12], "Annualized Volatility": [0.2]},
            index=pd.Index(["TCS.NS"], name="Stock"),
        )
        result = csv_bytes(computed).decode("utf-8-sig")
        self.assertIn("Stock,Expected Annualized Return,Annualized Volatility", result)
        self.assertIn("TCS.NS,0.12,0.2", result)
        self.assertNotIn("password", result.casefold())


if __name__ == "__main__":
    unittest.main()
