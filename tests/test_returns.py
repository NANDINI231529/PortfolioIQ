import unittest

import numpy as np
import pandas as pd

from analytics.returns import annualized_arithmetic_return, annualized_geometric_return, calculate_daily_returns, correlation_matrix, covariance_matrix, performance_metrics
from analytics.validation import ValidationError


class ReturnAnalyticsTests(unittest.TestCase):
    def setUp(self):
        # Synthetic fixture for offline tests only; this is not market data.
        index = pd.date_range("2024-01-01", periods=5, freq="B")
        self.synthetic_prices = pd.DataFrame({"AAA": [100, 101, 100, 104, 106], "BBB": [50, 49, 51, 52, 54]}, index=index)

    def test_daily_simple_returns(self):
        returns = calculate_daily_returns(self.synthetic_prices)
        self.assertAlmostEqual(returns.iloc[0]["AAA"], 0.01)
        self.assertAlmostEqual(returns.iloc[1]["AAA"], 100 / 101 - 1)
        self.assertEqual(len(returns), 4)

    def test_annualized_return_and_metrics(self):
        values = pd.Series([0.01, -0.005, 0.02])
        self.assertAlmostEqual(annualized_arithmetic_return(values), values.mean() * 252)
        expected_geo = np.prod(1 + values.to_numpy()) ** (252 / len(values)) - 1
        self.assertAlmostEqual(annualized_geometric_return(values), expected_geo)
        returns = calculate_daily_returns(self.synthetic_prices)
        metrics = performance_metrics(returns)
        self.assertAlmostEqual(metrics.loc["AAA", "Daily Variance"], returns["AAA"].var(ddof=1))
        self.assertAlmostEqual(metrics.loc["AAA", "Annualized Variance"], returns["AAA"].var(ddof=1) * 252)
        self.assertAlmostEqual(metrics.loc["AAA", "Daily Volatility"], returns["AAA"].std(ddof=1))
        self.assertAlmostEqual(metrics.loc["AAA", "Annualized Volatility"], returns["AAA"].std(ddof=1) * np.sqrt(252))

    def test_covariance_and_correlation(self):
        returns = calculate_daily_returns(self.synthetic_prices)
        daily_cov = covariance_matrix(returns)
        annual_cov = covariance_matrix(returns, annualized=True)
        corr = correlation_matrix(returns)
        self.assertEqual(list(daily_cov.columns), ["AAA", "BBB"])
        np.testing.assert_allclose(annual_cov.to_numpy(), daily_cov.to_numpy() * 252)
        self.assertAlmostEqual(corr.loc["AAA", "AAA"], 1.0)
        self.assertTrue(np.allclose(corr, corr.T, equal_nan=True))

    def test_missing_prices_do_not_bridge_gaps(self):
        prices = self.synthetic_prices.copy()
        prices.loc[prices.index[2], "AAA"] = np.nan
        returns = calculate_daily_returns(prices)
        self.assertTrue(np.isnan(returns.loc[prices.index[2], "AAA"]))
        self.assertTrue(np.isnan(returns.loc[prices.index[3], "AAA"]))
        self.assertTrue(np.isfinite(returns.loc[prices.index[4], "AAA"]))

    def test_invalid_and_insufficient_data(self):
        with self.assertRaises(ValidationError):
            calculate_daily_returns(pd.DataFrame())
        one_return = pd.DataFrame({"AAA": [0.01], "BBB": [np.nan]})
        with self.assertRaises(ValidationError):
            covariance_matrix(one_return)
        self.assertTrue(np.isnan(annualized_geometric_return(pd.Series([-1.0, 0.1]))))


if __name__ == "__main__":
    unittest.main()
