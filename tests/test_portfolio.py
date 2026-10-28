import unittest

import numpy as np
import pandas as pd

from analytics.portfolio import OptimizationError, maximum_sharpe_portfolio, minimum_volatility_portfolio, portfolio_metrics, simulate_portfolios
from analytics.validation import ValidationError


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        # Synthetic return fixture for offline tests only; it is not actual performance.
        rng = np.random.default_rng(731)
        index = pd.date_range("2023-01-02", periods=180, freq="B")
        base = rng.normal(0.00035, 0.008, size=180)
        self.synthetic_returns = pd.DataFrame({
            "AAA": base,
            "BBB": rng.normal(0.00065, 0.012, size=180),
            "CCC": 0.45 * base + rng.normal(0.00015, 0.005, size=180),
        }, index=index)

    def test_random_simulations_are_reproducible_and_weights_sum_to_one(self):
        first = simulate_portfolios(self.synthetic_returns, 120, 12, 0.05)
        second = simulate_portfolios(self.synthetic_returns, 120, 12, 0.05)
        pd.testing.assert_frame_equal(first, second)
        weight_columns = [column for column in first if column.startswith("Weight:")]
        np.testing.assert_allclose(first[weight_columns].sum(axis=1), 1.0)
        self.assertEqual(len(first), 120)
        self.assertTrue((first["Annualized Volatility"] >= 0).all())

    def test_portfolio_metrics(self):
        returns = self.synthetic_returns.mean().to_numpy() * 252
        cov = self.synthetic_returns.cov().to_numpy() * 252
        weights = [0.2, 0.3, 0.5]
        actual = portfolio_metrics(weights, returns, cov, 0.04)
        self.assertAlmostEqual(actual["Expected Annualized Return"], np.dot(weights, returns))
        self.assertGreaterEqual(actual["Annualized Variance"], 0)
        self.assertAlmostEqual(actual["Sharpe Ratio"], (actual["Expected Annualized Return"] - 0.04) / actual["Annualized Volatility"])

    def test_minimum_volatility_optimization(self):
        weights, metrics = minimum_volatility_portfolio(self.synthetic_returns)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue((weights >= 0).all())
        self.assertTrue((weights <= 1).all())
        self.assertGreaterEqual(metrics["Annualized Volatility"], 0)

    def test_maximum_sharpe_optimization(self):
        weights, metrics = maximum_sharpe_portfolio(self.synthetic_returns, 0.03)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue((weights >= 0).all())
        self.assertTrue(np.isfinite(metrics["Sharpe Ratio"]))

    def test_invalid_risk_free_rate_weights_and_simulation_count(self):
        with self.assertRaises(ValidationError):
            maximum_sharpe_portfolio(self.synthetic_returns, float("nan"))
        with self.assertRaises(ValidationError):
            portfolio_metrics([0.8, 0.8], [0.1, 0.2], np.eye(2))
        with self.assertRaises(ValidationError):
            simulate_portfolios(self.synthetic_returns, 99, 1)

    def test_undefined_sharpe_and_optimizer_failure_handling(self):
        metrics = portfolio_metrics([1.0], [0.1], [[0.0]], 0.02)
        self.assertTrue(np.isnan(metrics["Sharpe Ratio"]))
        flat = pd.DataFrame({"FLAT": np.zeros(20)})
        with self.assertRaises(OptimizationError):
            maximum_sharpe_portfolio(flat, 0.0)


if __name__ == "__main__":
    unittest.main()
