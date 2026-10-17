"""Run the existing Random Forest pipeline with NSE universe + time-aware stages."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
UNIVERSE_FILE = RAW_DIR / "selected_universe.json"
PROCESSING_STATE_FILE = PROCESSED_DIR / "processing_state.json"
TRAINING_STATE_FILE = PROCESSED_DIR / "training_state.json"
MARKET_CLOSE_CUTOFF = time(17, 0)
GENERATED_SUFFIXES = (
    "_cleaned.csv", "_features.csv", "_model_data.csv",
    "_training_metrics.csv", "_backtest.csv",
)


def run_script(script, args=None):
    command = [sys.executable, str(SRC_DIR / script)]
    if args:
        command.extend(args)
    print(f"\\n{'=' * 72}\\nRUNNING {script}\\n{'=' * 72}", flush=True)
    subprocess.run(command, cwd=BASE_DIR, check=True)


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def selected_tickers():
    universe = load_json(UNIVERSE_FILE)
    if not isinstance(universe, dict) or not universe:
        raise FileNotFoundError(f"Selected NSE universe missing or empty: {UNIVERSE_FILE}")
    return sorted(ticker.replace(".NS", "") for ticker in universe)


def latest_date_in_csv(path):
    try:
        data = pd.read_csv(path, usecols=["Date"])
        dates = pd.to_datetime(data["Date"], errors="coerce").dropna()
        return dates.max().date() if not dates.empty else None
    except (ValueError, OSError, pd.errors.ParserError):
        return None


def latest_raw_market_date(stocks):
    dates = []
    for stock in stocks:
        value = latest_date_in_csv(RAW_DIR / f"{stock}.csv")
        if value:
            dates.append(value)
    return max(dates).isoformat() if dates else None


def outputs_exist(stocks):
    return all(
        (PROCESSED_DIR / f"{stock}{suffix}").exists()
        for stock in stocks
        for suffix in ("_cleaned.csv", "_features.csv", "_model_data.csv")
    )


def remove_stale_files(stocks):
    """Keep generated files/models aligned to the selected universe."""
    selected = set(stocks)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for path in RAW_DIR.glob("*.csv"):
        if path.stem not in selected:
            path.unlink(missing_ok=True)

    for path in PROCESSED_DIR.iterdir():
        if not path.is_file():
            continue
        for suffix in GENERATED_SUFFIXES:
            if path.name.endswith(suffix):
                symbol = path.name[:-len(suffix)]
                if symbol not in selected:
                    path.unlink(missing_ok=True)
                break

    for path in MODEL_DIR.glob("*_final_model.joblib"):
        if path.name.removesuffix("_final_model.joblib") not in selected:
            path.unlink(missing_ok=True)


def should_process(stocks, raw_date, period):
    state = load_json(PROCESSING_STATE_FILE)
    if state.get("selected_stocks") != stocks:
        return True
    if state.get("latest_raw_market_data_date") != raw_date:
        return True
    if state.get("period") != period:
        return True
    if not outputs_exist(stocks):
        return True
    print("Selected stocks and market data are unchanged; skipping cleaning, features and target creation.")
    return False


def should_train(market_date, stocks, period):
    state = load_json(TRAINING_STATE_FILE)
    same_inputs_already_trained = (
        state.get("selected_stocks") == stocks
        and state.get("latest_market_data_date") == market_date
        and state.get("period") == period
        and state.get("training_completed") is True
    )
    # Some symbols can legitimately be skipped for insufficient history. Don't
    # repeatedly retrain the entire universe for those same inputs.
    if same_inputs_already_trained:
        print("Training already completed for these stocks/data; reusing saved models.")
        return False

    after_close = datetime.now().time() >= MARKET_CLOSE_CUTOFF

    # First run / changed universe: train immediately so predictions can be produced.
    if not state or state.get("selected_stocks") != stocks or state.get("period") != period:
        print("Initial training required, history period changed, or selected universe changed.")
        return True

    if not after_close:
        print("Before 5 PM: reusing saved models; training waits until after market close.")
        return False

    if state.get("latest_market_data_date") != market_date:
        print("After 5 PM and market data changed: retraining.")
        return True

    print("No new market-data date; skipping unnecessary training.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Run the NSE Random Forest stock analyzer.")
    parser.add_argument("-u", "--universe", choices=["nifty50", "nifty100", "nifty200", "nifty500", "all"], default="nifty50")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of NSE stocks")
    parser.add_argument("--period", default="2y", help="Yahoo Finance history period")
    parser.add_argument("--force-refresh", action="store_true", help="Force fresh market-data downloads")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Fetcher downloads from a dynamic NSE constituent list; cache avoids repeated downloads.
    run_script("data_loader.py", [
        "--universe", args.universe,
        "--period", args.period,
        *([] if args.limit is None else ["--limit", str(args.limit)]),
        *(["--force-refresh"] if args.force_refresh else []),
    ])

    stocks = selected_tickers()
    remove_stale_files(stocks)
    raw_date = latest_raw_market_date(stocks)
    if not raw_date:
        raise RuntimeError("No raw market data is available. Check the network and NSE/Yahoo Finance access.")

    if should_process(stocks, raw_date, args.period):
        # Preserve the existing cleaner, features and target logic exactly.
        for stage in ("data_cleaner.py", "feature_engineering.py", "create_target.py"):
            run_script(stage)
        save_json(PROCESSING_STATE_FILE, {
            "processing_calendar_date": datetime.now().date().isoformat(),
            "latest_raw_market_data_date": raw_date,
            "selected_stocks": stocks,
            "period": args.period,
        })

    market_date = raw_date
    if should_train(market_date, stocks, args.period):
        run_script("model_training.py")
        save_json(TRAINING_STATE_FILE, {
            "training_calendar_date": datetime.now().date().isoformat(),
            "latest_market_data_date": market_date,
            "selected_stocks": stocks,
            "period": args.period,
            "trained_after_close": datetime.now().time() >= MARKET_CLOSE_CUTOFF,
            "training_completed": True,
        })
    else:
        print("Using previously trained Random Forest models.")

    # Predictions always refresh, even if preprocessing/training were skipped.
    run_script("prediction.py")
    print("\\nDONE. Open data/processed/latest_predictions.csv")


if __name__ == "__main__":
    main()
