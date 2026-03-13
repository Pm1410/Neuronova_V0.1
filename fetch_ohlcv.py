"""
data/fetch_ohlcv.py
-------------------
Fetches historical OHLCV data from any CCXT-supported exchange
and saves as Parquet files. Also handles incremental updates.

Usage:
    python -m data.fetch_ohlcv --symbol BTC/USDT --timeframe 1h --years 3
"""

import os
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import ccxt
import pandas as pd
import numpy as np
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console

console = Console()


class OHLCVFetcher:
    """
    Fetches OHLCV candlestick data using CCXT.
    Supports pagination for long historical windows.
    Saves to Parquet for fast I/O.
    """

    TIMEFRAME_MS = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }

    def __init__(self, exchange_id: str = "bybit", raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Initialize exchange — no auth needed for public OHLCV
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "linear"},  # USDT perpetuals
            }
        )
        logger.info(f"Initialized exchange: {exchange_id}")

    def fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        years: float = 3,
        since: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch full OHLCV history for a symbol.

        Args:
            symbol:    e.g. "BTC/USDT"
            timeframe: e.g. "1h"
            years:     How many years back to fetch
            since:     Override start datetime

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        if since is None:
            since = datetime.utcnow() - timedelta(days=int(years * 365))

        since_ms = int(since.timestamp() * 1000)
        tf_ms = self.TIMEFRAME_MS[timeframe]
        all_candles = []

        console.print(
            f"\n[bold cyan]Fetching[/bold cyan] {symbol} | "
            f"{timeframe} | {years}y history from {since.date()}"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed} candles"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading {symbol}", total=None)

            cursor = since_ms
            while True:
                try:
                    candles = self.exchange.fetch_ohlcv(
                        symbol, timeframe, since=cursor, limit=1000
                    )
                except ccxt.RateLimitExceeded:
                    logger.warning("Rate limit hit — sleeping 10s")
                    time.sleep(10)
                    continue
                except ccxt.NetworkError as e:
                    logger.error(f"Network error: {e} — retrying in 5s")
                    time.sleep(5)
                    continue

                if not candles:
                    break

                all_candles.extend(candles)
                progress.update(task, completed=len(all_candles))

                last_ts = candles[-1][0]
                if last_ts >= int(datetime.utcnow().timestamp() * 1000) - tf_ms:
                    break  # caught up to present

                cursor = last_ts + tf_ms
                time.sleep(self.exchange.rateLimit / 1000)

        df = self._to_dataframe(all_candles)
        df = self._validate_and_clean(df)

        out_path = self._save(df, symbol, timeframe)
        console.print(
            f"[bold green]✓[/bold green] Saved {len(df):,} candles → {out_path}\n"
        )
        return df

    def fetch_incremental(self, symbol: str, timeframe: str = "1h") -> pd.DataFrame:
        """
        Load existing data and fetch only missing candles up to now.
        Perfect for daily scheduled updates.
        """
        path = self._get_path(symbol, timeframe)
        if not path.exists():
            logger.warning("No existing data found — running full fetch")
            return self.fetch(symbol, timeframe)

        existing = pd.read_parquet(path)
        last_ts = existing.index[-1]
        since = last_ts + pd.Timedelta(timeframe)

        logger.info(f"Incremental fetch from {since} for {symbol}")
        new_data = self.fetch(symbol, timeframe, since=since.to_pydatetime())

        combined = pd.concat([existing, new_data]).drop_duplicates()
        combined.sort_index(inplace=True)
        combined.to_parquet(path)
        logger.info(f"Updated {symbol} data: {len(combined):,} total candles")
        return combined

    # ─── Private Helpers ───────────────────────────────────────────────────

    def _to_dataframe(self, candles: list) -> pd.DataFrame:
        df = pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        return df

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        original_len = len(df)

        # Remove exact duplicate timestamps
        df = df[~df.index.duplicated(keep="last")]

        # Remove zero/negative OHLCV values
        df = df[(df[["open", "high", "low", "close", "volume"]] > 0).all(axis=1)]

        # Basic OHLC sanity: high >= low
        df = df[df["high"] >= df["low"]]

        # Flag and forward-fill any remaining NaNs
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values — forward-filling")
            df.ffill(inplace=True)

        dropped = original_len - len(df)
        if dropped > 0:
            logger.info(f"Removed {dropped} bad candles during validation")

        df.sort_index(inplace=True)
        return df

    def _get_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.raw_dir / f"{safe_symbol}_{timeframe}.parquet"

    def _save(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
        path = self._get_path(symbol, timeframe)
        df.to_parquet(path, compression="snappy")
        return path


def load_ohlcv(symbol: str, timeframe: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Convenience loader for downstream modules."""
    safe_symbol = symbol.replace("/", "_")
    path = Path(raw_dir) / f"{safe_symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No data found at {path}. Run: python main.py fetch --symbol {symbol}"
        )
    return pd.read_parquet(path)


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV data via CCXT")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--years", type=float, default=3)
    parser.add_argument("--exchange", default="bybit")
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    fetcher = OHLCVFetcher(exchange_id=args.exchange)

    if args.incremental:
        fetcher.fetch_incremental(args.symbol, args.timeframe)
    else:
        fetcher.fetch(args.symbol, args.timeframe, years=args.years)
