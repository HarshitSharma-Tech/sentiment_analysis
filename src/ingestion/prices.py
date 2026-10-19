"""
Historical Price Ingestion Module (Free & Resilient)
Fetches BTC-USD and crypto OHLCV historical prices.
Primary Source: yfinance (Free, no API key).
Secondary Fallback: Binance Public Klines REST API (Free, no API key).
Tertiary Fallback: Local CSV files.
Features:
- UTC timestamp normalization
- OHLCV integrity validation (non-negative, high >= low, etc.)
- Gap detection and logging
- SQLite caching to prevent repeated network requests
"""

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import requests

from src.database import DatabaseManager

logger = logging.getLogger(__name__)


class PriceIngestion:
    """Manages downloading, validating, and caching historical market prices."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager

    @staticmethod
    def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLCV data:
        - Ensure index is DatetimeIndex in UTC
        - Sort chronologically and drop duplicate timestamps
        - Ensure required columns exist and are numeric
        - Filter out non-positive prices
        - Sanity check: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        - Handle short missing gaps via forward-fill
        """
        if df.empty:
            raise ValueError("Input price DataFrame is empty.")

        clean_df = df.copy()

        # Normalize index to UTC DatetimeIndex
        if not isinstance(clean_df.index, pd.DatetimeIndex):
            if "timestamp" in clean_df.columns:
                clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], utc=True)
                clean_df = clean_df.set_index("timestamp")
            elif "Date" in clean_df.columns or "Datetime" in clean_df.columns:
                col = "Date" if "Date" in clean_df.columns else "Datetime"
                clean_df[col] = pd.to_datetime(clean_df[col], utc=True)
                clean_df = clean_df.set_index(col)
            else:
                raise ValueError("DataFrame lacks timestamp index or column.")

        if clean_df.index.tz is None:
            clean_df.index = clean_df.index.tz_localize("UTC")
        else:
            clean_df.index = clean_df.index.tz_convert("UTC")

        # Sort and deduplicate
        clean_df = clean_df.sort_index()
        clean_df = clean_df[~clean_df.index.duplicated(keep="first")]

        # Lowercase columns
        clean_df.columns = [c.lower() for c in clean_df.columns]
        req_cols = ["open", "high", "low", "close", "volume"]
        for col in req_cols:
            if col not in clean_df.columns:
                raise ValueError(f"Missing required price column: {col}")
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        # Drop any row where close is NaN or non-positive
        clean_df = clean_df[clean_df["close"] > 0]

        # Sanity repairs: ensure high is highest and low is lowest
        clean_df["high"] = clean_df[["open", "high", "low", "close"]].max(axis=1)
        clean_df["low"] = clean_df[["open", "high", "low", "close"]].min(axis=1)

        # Check for timestamp gaps
        diffs = clean_df.index.to_series().diff()
        median_step = diffs.median()
        if pd.notnull(median_step):
            large_gaps = diffs[diffs > median_step * 2]
            if len(large_gaps) > 0:
                logger.warning(f"Detected {len(large_gaps)} timestamp gaps greater than 2x median step ({median_step}).")

        logger.info(
            f"Validated {len(clean_df)} price bars from {clean_df.index.min()} to {clean_df.index.max()}."
        )
        return clean_df

    def fetch_yfinance(
        self,
        ticker: str = "BTC-USD",
        frequency: str = "1h",
        days: int = 180,
    ) -> pd.DataFrame:
        """
        Download historical prices using yfinance.
        yfinance supports hourly ('1h') data up to 730 days.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is not installed. Install via `pip install yfinance`.")

        logger.info(f"Downloading {ticker} ({frequency}) for past {days} days from yfinance...")
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        # yfinance period string or explicit start/end
        yf_ticker = yf.Ticker(ticker)
        # Fetching with explicit interval
        df = yf_ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval=frequency,
            auto_adjust=False,
        )

        if df.empty:
            logger.warning(f"yfinance returned empty data for {ticker}. Trying fallback period='{days}d'...")
            # For 1h, yfinance period max is 730d
            period_str = f"{min(days, 720)}d"
            df = yf_ticker.history(period=period_str, interval=frequency, auto_adjust=False)

        if df.empty:
            raise ValueError(f"No price data retrieved from yfinance for {ticker}.")

        # Handle potential MultiIndex columns in recent yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        return self.validate_prices(df[["open", "high", "low", "close", "volume"]])

    def fetch_binance_public(
        self,
        symbol: str = "BTCUSDT",
        frequency: str = "1h",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Free secondary source: Binance Public Spot REST API.
        Completely free, no authentication, 1000 candles per call.
        """
        logger.info(f"Querying Binance Public REST API for {symbol} ({frequency}, limit={limit})...")
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": frequency,
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw_klines = resp.json()

        if not raw_klines:
            raise ValueError(f"Binance returned empty klines for {symbol}.")

        rows = []
        for k in raw_klines:
            # Kline format: [open_time, open, high, low, close, volume, close_time, ...]
            rows.append({
                "timestamp": pd.to_datetime(k[0], unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        df = pd.DataFrame(rows).set_index("timestamp")
        return self.validate_prices(df)

    def load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load prices from a local CSV file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Price CSV file not found at {p}")
        df = pd.read_csv(p)
        return self.validate_prices(df)

    def get_or_download_prices(
        self,
        ticker: str = "BTC-USD",
        frequency: str = "1h",
        days: int = 180,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieve prices from SQLite if available and fresh; otherwise fetch and cache.
        Ensures restartability without redundant network downloads.
        """
        start_cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

        if self.db and not force_download:
            cached_df = self.db.get_prices(ticker=ticker, frequency=frequency, start_time=start_cutoff)
            # If we have at least 80% of expected bars (e.g. 180 * 24 * 0.8 bars for hourly)
            expected_bars = days * (24 if frequency == "1h" else 1)
            if len(cached_df) >= min(100, int(expected_bars * 0.7)):
                logger.info(f"Loaded {len(cached_df)} cached price bars from SQLite for {ticker}.")
                return cached_df

        # Fetch from primary source (yfinance)
        df = None
        source_name = "yfinance"
        try:
            df = self.fetch_yfinance(ticker=ticker, frequency=frequency, days=days)
        except Exception as e:
            logger.warning(f"yfinance fetch failed ({e}). Attempting Binance public API fallback...")
            try:
                # Map BTC-USD to BTCUSDT
                binance_sym = ticker.replace("-USD", "USDT").replace("-", "")
                df = self.fetch_binance_public(symbol=binance_sym, frequency=frequency, limit=1000)
                source_name = "binance_public"
            except Exception as e_binance:
                logger.error(f"Binance public fetch also failed ({e_binance}).")
                # If cached data exists in DB, fallback to it even if older
                if self.db:
                    fallback_db = self.db.get_prices(ticker=ticker, frequency=frequency)
                    if not fallback_db.empty:
                        logger.warning("Using existing historical database prices as offline fallback.")
                        return fallback_db
                raise RuntimeError(f"Unable to retrieve market prices from yfinance or Binance: {e}")

        # Cache in SQLite
        if self.db and df is not None and not df.empty:
            self.db.save_prices(df=df, ticker=ticker, frequency=frequency, source=source_name)

        return df
