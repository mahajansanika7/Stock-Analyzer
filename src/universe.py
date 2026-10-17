"""Download NSE stock universes (cached for 7 days)."""
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
UNIVERSE_FILE = RAW_DATA_DIR / "selected_universe.json"

SOURCES = {
    "nifty50": [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    ],
    "nifty100": [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    ],
    "nifty200": [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    ],
    "nifty500": [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    ],
    "all": [
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    ],
}
# Used only if NSE cannot be reached and no cached universe exists.
FALLBACK_NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,*/*",
}


def _download_symbols(urls, equity_only):
    last_error = None
    for url in urls:
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            frame = pd.read_csv(io.StringIO(response.text))
            frame.columns = [str(c).strip().upper() for c in frame.columns]
            if "SYMBOL" not in frame.columns:
                raise ValueError("NSE CSV has no SYMBOL column")
            if equity_only and "SERIES" in frame.columns:
                frame = frame[frame["SERIES"].astype(str).str.strip().eq("EQ")]
            name_col = next((c for c in ("NAME OF COMPANY", "COMPANY NAME", "COMPANY") if c in frame.columns), None)
            names = {}
            for _, row in frame.iterrows():
                symbol = str(row["SYMBOL"]).strip()
                if not symbol or symbol.lower() == "nan":
                    continue
                company = str(row[name_col]).strip() if name_col else symbol
                if not company or company.lower() == "nan":
                    company = symbol
                names[symbol] = company
            if not names:
                raise ValueError("NSE CSV returned no symbols")
            return dict(sorted(names.items()))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not download NSE stock list: {last_error}")


def get_universe(name="nifty50", refresh_days=7):
    name = name.lower()
    if name not in SOURCES:
        raise ValueError(f"Unknown universe {name!r}. Choose: {', '.join(SOURCES)}")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DATA_DIR / f"universe_{name}.json"
    names = None
    if cache_path.exists() and time.time() - cache_path.stat().st_mtime < refresh_days * 86400:
        try:
            names = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            names = None
    if not names:
        try:
            names = _download_symbols(SOURCES[name], equity_only=(name == "all"))
            cache_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
            print(f"Loaded {len(names)} symbols from NSE universe: {name}")
        except Exception as exc:
            if cache_path.exists():
                print(f"NSE universe unavailable; using cached {name} list ({exc})")
                names = json.loads(cache_path.read_text(encoding="utf-8"))
            elif name == "nifty50":
                print(f"NSE universe unavailable; using fallback NIFTY 50 list ({exc})")
                names = {symbol: symbol for symbol in FALLBACK_NIFTY50}
            else:
                print(f"NSE universe unavailable and no cached {name} list exists: {exc}")
                raise
    # Store as Yahoo Finance symbols, matching the existing data pipeline.
    selected = {f"{symbol}.NS": company for symbol, company in names.items()}
    UNIVERSE_FILE.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected
