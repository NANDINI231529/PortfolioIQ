import unittest

import numpy as np
import pandas as pd

from analytics.validation import ValidationError, clean_price_data, validate_dates, validate_random_seed, validate_risk_free_rate, validate_simulation_count, validate_tickers, validate_weights


class ValidationTests(unittest.TestCase):
    def test_stock_and_date_inputs(self):
        self.assertEqual(validate_tickers([" tcs.ns ", "TCS.NS"]), ["TCS.NS"])
        for selected in ([], [" "]):
            with self.assertRaises(ValidationError):
                validate_tickers(selected)
        with self.assertRaises(ValidationError):
            validate_dates(pd.Timestamp("2024-02-01").date(), pd.Timestamp("2024-01-01").date())

    def test_simulation_seed_rate_and_weights(self):
        self.assertEqual(validate_simulation_count(500), 500)
        self.assertEqual(validate_random_seed(0), 0)
        self.assertAlmostEqual(validate_risk_free_rate(5.0), 0.05)
        np.testing.assert_allclose(validate_weights([0.25, 0.75]), [0.25, 0.75])
        for call in [lambda: validate_simulation_count(1), lambda: validate_random_seed(-1),
                     lambda: validate_risk_free_rate(float("inf")), lambda: validate_weights([0.4, 0.4])]:
            with self.assertRaises(ValidationError):
                call()

    def test_price_cleanup_handles_duplicates_invalid_and_non_numeric(self):
        index = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"])
        raw = pd.DataFrame({"AAA": [10, 9, 11], "BBB": ["bad", 20, "bad"]}, index=index)
        cleaned = clean_price_data(raw)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned.index.is_monotonic_increasing, True)
        self.assertEqual(cleaned.loc[pd.Timestamp("2024-01-02"), "AAA"], 11)
        self.assertTrue(np.isnan(cleaned.loc[pd.Timestamp("2024-01-02"), "BBB"]))


if __name__ == "__main__":
    unittest.main()
