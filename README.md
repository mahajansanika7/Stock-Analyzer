# NSE Random Forest Stock Analyzer

This project keeps the original Random Forest model, features, five-trading-day target,
walk-forward validation, backtest, probability calculation, and BUY CANDIDATE threshold.
The changes are limited to dynamic NSE universe selection, data-fetch caching, and
time-aware pipeline scheduling.

## Setup (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bat
python run_stock_analyzer.py
```

Default is the live NSE NIFTY 50 constituent list (with a cached/fallback list if NSE is
temporarily unavailable). Choose another NSE universe:

```bat
python src\run_predictions.py --universe nifty100
python src\run_predictions.py --universe nifty200
python src\run_predictions.py --universe nifty500
python src\run_predictions.py --universe all
python src\run_predictions.py --universe nifty50 --limit 20
python src\run_predictions.py --universe nifty50 --period 5y
python src\run_predictions.py --universe nifty50 --force-refresh
```

`run_stock_analyzer.py` uses the default NIFTY 50 universe. For options, run
`src\run_predictions.py` directly.

## Scheduling and cache behavior

- NSE constituent lists are cached for 7 days.
- Price history is cached until the next completed NSE market close (approximately 16:00 IST).
- During market hours, today's incomplete candle is excluded.
- Cleaning, feature engineering and target creation run only when the selected universe,
  latest raw market date, or required outputs change.
- The first run or a changed universe triggers initial training.
- Subsequent training is deferred until after 17:00 IST when market data has advanced.
- If training already completed for the same selected stocks and market-data date, it is not
  repeated, even if a few stocks were skipped for insufficient history.
- Predictions are refreshed on every run using the saved models.

## Model and data note

NSE supplies the dynamic stock constituent list; daily OHLC data is fetched through
Yahoo Finance (`yfinance`). The model is a screening/learning tool, not financial advice.
