"""Random long-only portfolio simulation and constrained optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .returns import TRADING_DAYS, aligned_returns
from .validation import ValidationError, validate_covariance, validate_risk_free_rate, validate_simulation_count, validate_weights


class OptimizationError(RuntimeError):
    """Portfolio optimization was unavailable or did not converge."""


def portfolio_metrics(weights, expected_returns, covariance, risk_free_rate: float = 0.0) -> dict[str, float]:
    w = validate_weights(weights)
    mu = np.asarray(expected_returns, dtype=float)
    cov = validate_covariance(covariance)
    if mu.ndim != 1 or len(mu) != len(w) or cov.shape != (len(w), len(w)):
        raise ValidationError("Weights, expected returns, and covariance matrix must have matching dimensions.")
    if not np.isfinite(mu).all():
        raise ValidationError("Expected returns contain invalid numerical values.")
    rf = float(risk_free_rate)
    if not np.isfinite(rf) or not -1 <= rf <= 1:
        raise ValidationError("Risk-free rate must be a finite decimal between -1 and 1.")
    annual_return = float(w @ mu)
    variance = float(w @ cov @ w)
    if variance < -1e-10:
        raise ValidationError("Portfolio variance is materially negative; the covariance inputs are invalid.")
    variance = max(variance, 0.0)
    volatility = float(np.sqrt(variance))
    sharpe = (annual_return - rf) / volatility if volatility > 1e-12 else float("nan")
    return {"Expected Annualized Return": annual_return, "Annualized Variance": variance,
            "Annualized Volatility": volatility, "Sharpe Ratio": float(sharpe)}


def simulate_portfolios(daily_returns: pd.DataFrame, count: int = 5_000, seed: int = 42, risk_free_rate: float = 0.0) -> pd.DataFrame:
    count = validate_simulation_count(count)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) <= 2**32 - 1:
        raise ValidationError("Random seed must be an integer from 0 to 4,294,967,295.")
    rf = float(risk_free_rate)
    if not np.isfinite(rf) or not -1 <= rf <= 1:
        raise ValidationError("Risk-free rate must be a finite decimal between -1 and 1.")
    aligned = aligned_returns(daily_returns)
    labels = list(aligned.columns)
    mu = aligned.mean().to_numpy(dtype=float) * TRADING_DAYS
    cov = aligned.cov().to_numpy(dtype=float) * TRADING_DAYS
    validate_covariance(cov)
    weights = np.random.default_rng(int(seed)).dirichlet(np.ones(len(labels)), size=count)
    portfolio_returns = weights @ mu
    variances = np.einsum("ij,jk,ik->i", weights, cov, weights)
    if (variances < -1e-10).any():
        raise ValidationError("Simulation produced invalid negative portfolio variance.")
    volatility = np.sqrt(np.maximum(variances, 0.0))
    sharpe = np.divide(portfolio_returns - rf, volatility, out=np.full(count, np.nan), where=volatility > 1e-12)
    result = pd.DataFrame(weights, columns=[f"Weight: {label}" for label in labels])
    result.insert(0, "Portfolio ID", np.arange(1, count + 1))
    result["Expected Annualized Return"] = portfolio_returns
    result["Annualized Volatility"] = volatility
    result["Sharpe Ratio"] = sharpe
    return result


def _optimize(daily_returns: pd.DataFrame, risk_free_rate: float, objective: str) -> tuple[pd.Series, dict[str, float]]:
    aligned = aligned_returns(daily_returns)
    labels = list(aligned.columns)
    mu = aligned.mean().to_numpy(dtype=float) * TRADING_DAYS
    cov = aligned.cov().to_numpy(dtype=float) * TRADING_DAYS
    validate_covariance(cov)
    if not np.isfinite(mu).all() or len(mu) == 0:
        raise OptimizationError("Historical return estimates are not defined for this selection.")
    rf = float(risk_free_rate)
    if not np.isfinite(rf) or not -1 <= rf <= 1:
        raise ValidationError("Risk-free rate must be a finite decimal between -1 and 1.")
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise OptimizationError("SciPy is required for portfolio optimization. Install the project requirements.") from exc
    initial = np.full(len(labels), 1 / len(labels))

    def variance_objective(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    def sharpe_objective(weights: np.ndarray) -> float:
        metrics = portfolio_metrics(weights, mu, cov, rf)
        if not np.isfinite(metrics["Sharpe Ratio"]):
            return 1e6
        return -metrics["Sharpe Ratio"]

    fun = variance_objective if objective == "minimum_volatility" else sharpe_objective
    result = minimize(fun, initial, method="SLSQP", bounds=[(0.0, 1.0)] * len(labels),
                      constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1)},
                      options={"maxiter": 2_000, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise OptimizationError(f"The {objective.replace('_', ' ')} optimizer did not converge: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    if (weights < -1e-6).any() or (weights > 1 + 1e-6).any() or abs(weights.sum() - 1) > 1e-5:
        raise OptimizationError("The optimizer returned weights that do not satisfy long-only constraints.")
    weights = np.clip(weights, 0, 1)
    weights /= weights.sum()
    metrics = portfolio_metrics(weights, mu, cov, rf)
    if objective == "maximum_sharpe" and not np.isfinite(metrics["Sharpe Ratio"]):
        raise OptimizationError("A maximum-Sharpe portfolio is undefined because volatility is zero or invalid.")
    return pd.Series(weights, index=labels, name="Weight"), metrics


def minimum_volatility_portfolio(daily_returns: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    return _optimize(daily_returns, 0.0, "minimum_volatility")


def maximum_sharpe_portfolio(daily_returns: pd.DataFrame, risk_free_rate: float = 0.0) -> tuple[pd.Series, dict[str, float]]:
    return _optimize(daily_returns, risk_free_rate, "maximum_sharpe")
