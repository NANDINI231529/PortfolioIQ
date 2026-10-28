"""Centralized input and numerical validation helpers."""

from __future__ import annotations

from datetime import date
import math
from typing import Iterable

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when an analysis input cannot be used safely."""


def validate_tickers(tickers: Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        tickers = [tickers]
    cleaned = list(dict.fromkeys(t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()))
    if not cleaned:
        raise ValidationError("Select at least one stock.")
    if any(len(t) > 30 or any(c.isspace() for c in t) for t in cleaned):
        raise ValidationError("One or more stock symbols are invalid.")
    return cleaned


def validate_dates(start: date, end: date) -> None:
    if start is None or end is None:
        raise ValidationError("Choose both a start date and an end date.")
    if start >= end:
        raise ValidationError("The start date must be earlier than the end date.")


def validate_simulation_count(count: int, minimum: int = 100, maximum: int = 50_000) -> int:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not minimum <= int(count) <= maximum:
        raise ValidationError(f"Simulation count must be between {minimum:,} and {maximum:,}.")
    return int(count)


def validate_random_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) <= 2**32 - 1:
        raise ValidationError("Random seed must be an integer from 0 to 4,294,967,295.")
    return int(seed)


def validate_risk_free_rate(rate: float) -> float:
    try:
        value = float(rate)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Enter a valid annual risk-free rate.") from exc
    if not math.isfinite(value) or not -100 <= value <= 100:
        raise ValidationError("Risk-free rate must be between -100% and 100%.")
    return value / 100.0


def clean_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValidationError("No historical price data is available for the selected stocks.")
    result = prices.copy()
    if isinstance(result.columns, pd.MultiIndex):
        raise ValidationError("Price data must have one column per stock after normalization.")
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result.loc[~result.index.isna()]
    result = result[~result.index.duplicated(keep="last")].sort_index()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.where(result > 0)
    result = result.dropna(axis=1, how="all")
    if result.empty or result.shape[1] == 0:
        raise ValidationError("The downloaded data contains no valid positive prices.")
    return result


def validate_weights(weights: Iterable[float], tolerance: float = 1e-6) -> np.ndarray:
    array = np.asarray(list(weights), dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValidationError("Portfolio weights must be a non-empty vector of finite numbers.")
    if (array < -tolerance).any() or (array > 1 + tolerance).any() or abs(float(array.sum()) - 1) > tolerance:
        raise ValidationError("Long-only portfolio weights must be between 0 and 1 and sum to 1.")
    return np.clip(array, 0.0, 1.0)


def validate_covariance(covariance: pd.DataFrame | np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValidationError("Covariance matrix must be square and non-empty.")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValidationError("Covariance matrix contains invalid or inconsistent values.")
    if (np.diag(matrix) < -1e-10).any():
        raise ValidationError("Covariance matrix cannot have negative variances.")
    return (matrix + matrix.T) / 2
