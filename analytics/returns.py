"""Daily return, annualization, covariance, and correlation calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import ValidationError, clean_price_data

TRADING_DAYS = 252


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    clean = clean_price_data(prices)
    # fill_method=None prevents bridging an unavailable observation with an older price.
    returns = clean.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return returns.dropna(axis=0, how="all")


def annualized_arithmetic_return(daily_returns: pd.Series | pd.DataFrame, trading_days: int = TRADING_DAYS):
    if trading_days <= 0:
        raise ValidationError("Trading days per year must be positive.")
    values = daily_returns.replace([np.inf, -np.inf], np.nan)
    result = values.mean(skipna=True) * trading_days
    return result


def annualized_geometric_return(daily_returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    values = pd.to_numeric(daily_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0 or (values <= -1).any() or trading_days <= 0:
        return float("nan")
    return float(np.expm1(np.log1p(values.to_numpy(dtype=float)).sum() * trading_days / len(values)))


def performance_metrics(daily_returns: pd.DataFrame, trading_days: int = TRADING_DAYS) -> pd.DataFrame:
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty:
        raise ValidationError("At least two valid daily returns are required.")
    records: list[dict[str, float | int | str]] = []
    for ticker in daily_returns.columns:
        series = pd.to_numeric(daily_returns[ticker], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        variance = float(series.var(ddof=1)) if len(series) >= 2 else float("nan")
        mean = float(series.mean()) if len(series) else float("nan")
        records.append({
            "Stock": str(ticker), "Observations": int(len(series)),
            "Latest Daily Return": float(series.iloc[-1]) if len(series) else float("nan"),
            "Mean Daily Return": mean,
            "Annualized Arithmetic Return": mean * trading_days if len(series) else float("nan"),
            "Annualized Geometric Return": annualized_geometric_return(series, trading_days),
            "Daily Variance": variance,
            "Annualized Variance": variance * trading_days if np.isfinite(variance) else float("nan"),
            "Daily Volatility": float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0 else float("nan"),
            "Annualized Volatility": float(np.sqrt(variance * trading_days)) if np.isfinite(variance) and variance >= 0 else float("nan"),
        })
    return pd.DataFrame.from_records(records).set_index("Stock")


def aligned_returns(daily_returns: pd.DataFrame, minimum_observations: int = 2) -> pd.DataFrame:
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty:
        raise ValidationError("No daily return observations are available.")
    aligned = daily_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(aligned) < minimum_observations:
        raise ValidationError("Selected stocks do not have enough overlapping return observations.")
    return aligned


def covariance_matrix(daily_returns: pd.DataFrame, annualized: bool = False, trading_days: int = TRADING_DAYS) -> pd.DataFrame:
    aligned = aligned_returns(daily_returns)
    cov = aligned.cov(ddof=1)
    if annualized:
        cov = cov * trading_days
    return cov


def correlation_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty:
        raise ValidationError("No daily return observations are available for correlation analysis.")
    result = daily_returns.replace([np.inf, -np.inf], np.nan).corr(method="pearson", min_periods=2)
    return result
