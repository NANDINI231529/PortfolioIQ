import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from analytics.market_data import MarketDataError, _extract_price_frame, download_prices


class MarketDataStructureTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2024-01-01", periods=3, freq="B")

    def test_field_first_multiindex_uses_adjusted_close(self):
        columns = pd.MultiIndex.from_product([["Adj Close", "Close"], ["AAA", "BBB"]])
        raw = pd.DataFrame([[10, 20, 11, 21], [11, 21, 12, 22], [12, 22, 13, 23]], index=self.index, columns=columns)
        prices, basis = _extract_price_frame(raw, ["AAA", "BBB"])
        self.assertEqual(basis, "Adj Close")
        self.assertEqual(list(prices.columns), ["AAA", "BBB"])
        self.assertEqual(prices.iloc[0].tolist(), [10, 20])

    def test_ticker_first_multiindex_and_close_fallback(self):
        columns = pd.MultiIndex.from_product([["AAA", "BBB"], ["Open", "Close"]])
        raw = pd.DataFrame([[8, 10, 18, 20], [9, 11, 19, 21], [10, 12, 20, 22]], index=self.index, columns=columns)
        prices, basis = _extract_price_frame(raw, ["AAA", "BBB"])
        self.assertEqual(basis, "Close")
        self.assertEqual(prices.iloc[0].tolist(), [10, 20])

    def test_missing_price_field_or_symbol_fails_clearly(self):
        raw = pd.DataFrame({"Open": [1, 2, 3]}, index=self.index)
        with self.assertRaises(MarketDataError):
            _extract_price_frame(raw, ["AAA"])
        raw = pd.DataFrame({"Close": [1, 2, 3]}, index=self.index)
        with self.assertRaises(MarketDataError):
            _extract_price_frame(raw, ["AAA", "BBB"])

    def test_download_network_failure_becomes_safe_market_data_error(self):
        with patch("yfinance.download", side_effect=ConnectionError("offline fixture")):
            with self.assertRaisesRegex(MarketDataError, "Market data could not be retrieved"):
                download_prices(["TCS.NS"], date(2024, 1, 1), date(2024, 2, 1))


if __name__ == "__main__":
    unittest.main()
