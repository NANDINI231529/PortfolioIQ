"""Yahoo Finance download and schema normalization."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from .validation import ValidationError, clean_price_data, validate_dates, validate_tickers

logger = logging.getLogger(__name__)


class MarketDataError(RuntimeError):
    """A market download failed or returned unusable prices."""


def _extract_price_frame(raw: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, str]:
    if raw is None or raw.empty:
        raise MarketDataError("Yahoo Finance returned no data for this selection and period.")
    if isinstance(raw.columns, pd.MultiIndex):
        level_values = [set(map(str, raw.columns.get_level_values(i))) for i in range(raw.columns.nlevels)]
        price_level = next((i for i, values in enumerate(level_values) if "Adj Close" in values or "Close" in values), None)
        if price_level is None:
            raise MarketDataError("The download did not include an adjusted-close or close price field.")
        price_field = "Adj Close" if "Adj Close" in level_values[price_level] else "Close"
        sliced = raw.xs(price_field, axis=1, level=price_level, drop_level=True)
        if isinstance(sliced, pd.Series):
            sliced = sliced.to_frame(name=tickers[0])
        if isinstance(sliced.columns, pd.MultiIndex):
            sliced.columns = [next((str(part) for part in col if str(part) in tickers), str(col[-1])) for col in sliced.columns]
        frame = sliced
    else:
        price_field = "Adj Close" if "Adj Close" in raw.columns else "Close" if "Close" in raw.columns else None
        if price_field is None:
            raise MarketDataError("The download did not include an adjusted-close or close price field.")
        frame = raw[[price_field]].rename(columns={price_field: tickers[0]}) if len(tickers) == 1 else raw
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=tickers[0])
    missing = [ticker for ticker in tickers if ticker not in frame.columns]
    if missing:
        raise MarketDataError(f"No usable price series was returned for: {', '.join(missing)}.")
    frame = frame.loc[:, tickers]
    return frame, price_field


def download_prices(tickers: list[str], start: date, end: date) -> tuple[pd.DataFrame, str]:
    symbols = validate_tickers(tickers)
    validate_dates(start, end)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise MarketDataError("yfinance is not installed. Install the requirements to download market data.") from exc
    try:
        # Yahoo's end date is exclusive, so add one day to include the selected end date.
        raw = yf.download(
            tickers=symbols, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=False, progress=False, group_by="column", threads=False, timeout=20,
        )
        prices, price_field = _extract_price_frame(raw, symbols)
        prices = clean_price_data(prices)
        absent = [symbol for symbol in symbols if symbol not in prices.columns]
        if absent:
            raise MarketDataError(f"No valid prices were returned for: {', '.join(absent)}.")
        if prices.shape[0] < 3:
            raise MarketDataError("At least three valid price observations are needed for return analysis.")
        if price_field == "Close":
            logger.info("Adjusted Close was unavailable; using Close for selected market data.")
        return prices, "Adjusted Close" if price_field == "Adj Close" else "Close (adjusted data unavailable)"
    except MarketDataError:
        raise
    except Exception as exc:
        logger.warning("Yahoo Finance download failed (%s).", type(exc).__name__)
        raise MarketDataError("Market data could not be retrieved. Check the connection, symbols, and date range, then try again.") from exc
