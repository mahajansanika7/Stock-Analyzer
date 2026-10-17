"""NSE universe selection and market-aware Yahoo Finance data downloads.

The model pipeline is unchanged. Only the stock universe and data-fetch/cache
behaviour are managed here.
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from universe import get_universe

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
UNIVERSE_FILE = RAW_DATA_DIR / "selected_universe.json"
DOWNLOAD_STATE_FILE = RAW_DATA_DIR / "download_state.json"
MARKET_TZ_NAME = "Asia/Kolkata"
MARKET_OPEN = "09:15"
DATA_READY_TIME = "16:00"


def _timezone():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(MARKET_TZ_NAME)
    except Exception:
        from datetime import timezone
        return timezone(timedelta(hours=5, minutes=30))


TZ = _timezone()


def _now():
    return datetime.now(TZ)


def last_data_ready_time(now=None):
    """Most recent weekday data-ready cutoff (16:00 IST)."""
    now = now or _now()
    ready = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now < ready:
        ready -= timedelta(days=1)
    while ready.weekday() >= 5:
        ready -= timedelta(days=1)
    return ready


def market_is_open(now=None):
    now = now or _now()
    if now.weekday() >= 5:
        return False
    opening = now.replace(hour=9, minute=15, second=0, microsecond=0)
    ready = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return opening <= now < ready


def _cache_is_fresh(path):
    if not path.exists():
        return False
    saved = datetime.fromtimestamp(path.stat().st_mtime, TZ)
    return saved >= last_data_ready_time()


def _flatten_yfinance_columns(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def _download_one(ticker, period):
    data = yf.download(
        ticker, period=period, interval="1d", auto_adjust=False,
        progress=False, actions=True, threads=False,
    )
    if data is None or data.empty:
        raise ValueError("No data returned")
    data = _flatten_yfinance_columns(data).reset_index()
    data.columns = [str(c) for c in data.columns]
    if "Date" not in data.columns:
        data.rename(columns={data.columns[0]: "Date"}, inplace=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")
    # Don't retain today's unfinished candle during the trading session.
    if market_is_open() and not data.empty and data["Date"].iloc[-1].date() == _now().date():
        data = data.iloc[:-1]
    return data


def download_stock_data(universe="nifty50", period="2y", limit=None, force_refresh=False):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stocks = get_universe(universe)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        stocks = dict(list(stocks.items())[:limit])
        UNIVERSE_FILE.write_text(json.dumps(stocks, indent=2), encoding="utf-8")
    print(f"Selected NSE universe {universe}: {len(stocks)} stocks")
    try:
        previous_state = json.loads(DOWNLOAD_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous_state = {}
    if previous_state.get("period") != period:
        force_refresh = True
        print(f"History period changed to {period}; refreshing downloaded price history.")

    succeeded = 0
    failed = []
    for ticker, company in stocks.items():
        output_file = RAW_DATA_DIR / f"{ticker.replace('.NS', '')}.csv"
        if not force_refresh and _cache_is_fresh(output_file):
            succeeded += 1
            continue
        try:
            print(f"Downloading {company} ({ticker})...")
            data = _download_one(ticker, period)
            if data.empty:
                raise ValueError("No completed daily candles returned")
            data.to_csv(output_file, index=False)
            succeeded += 1
        except Exception as exc:
            failed.append((ticker, str(exc)))
            # Keep a previous usable cache file if one exists.
            if not output_file.exists():
                print(f"  Failed: {exc}")
    DOWNLOAD_STATE_FILE.write_text(json.dumps({
        "universe": universe,
        "period": period,
        "selected_tickers": list(stocks),
        "last_fetch_attempt": datetime.now(TZ).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"Market data ready for {succeeded}/{len(stocks)} stocks; failed/newly unavailable: {len(failed)}")
    for ticker, reason in failed[:10]:
        print(f"  {ticker}: {reason}")
    return stocks


def main():
    parser = argparse.ArgumentParser(description="Fetch market data for a dynamic NSE universe.")
    parser.add_argument("-u", "--universe", choices=["nifty50", "nifty100", "nifty200", "nifty500", "all"], default="nifty50")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of NSE symbols")
    parser.add_argument("--force-refresh", action="store_true", help="Force fresh Yahoo Finance downloads")
    args = parser.parse_args()
    download_stock_data(args.universe, args.period, args.limit, args.force_refresh)


if __name__ == "__main__":
    main()
