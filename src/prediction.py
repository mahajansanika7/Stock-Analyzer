from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FEATURE_COLUMNS = ["Daily_Return", "Return_5D", "Return_20D", "MA_20_Ratio", "MA_50_Ratio", "Volatility_20D", "RSI_14"]
BUY_THRESHOLD = 0.55  # model probability threshold for a candidate, not a guarantee
UNIVERSE_FILE = BASE_DIR / "data" / "raw" / "selected_universe.json"


def load_selected_universe():
    """Load the current NSE universe chosen by data_loader.py."""
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Selected universe is missing: {UNIVERSE_FILE}. Run the pipeline first."
        )
    with UNIVERSE_FILE.open("r", encoding="utf-8") as handle:
        universe = json.load(handle)
    if not isinstance(universe, dict) or not universe:
        raise ValueError(f"Selected universe is empty or invalid: {UNIVERSE_FILE}")
    return universe

def predict_stock(ticker: str, company_name: str):
    stock = ticker.replace(".NS", "")
    feature_path = PROCESSED_DATA_DIR / f"{stock}_features.csv"
    model_path = MODEL_DIR / f"{stock}_final_model.joblib"
    if not feature_path.exists() or not model_path.exists():
        print(f"Skipping {ticker}: feature file or trained model is missing")
        return None
    data = pd.read_csv(feature_path)
    missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing or data.empty:
        print(f"Skipping {ticker}: missing features {missing} or empty file")
        return None
    row = data.iloc[[-1]].copy()
    row[FEATURE_COLUMNS] = row[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if row[FEATURE_COLUMNS].isna().any().any():
        print(f"Skipping {ticker}: latest feature row has missing values")
        return None
    model = joblib.load(model_path)
    probs = model.predict_proba(row[FEATURE_COLUMNS])[0]
    positive_probability = float(probs[list(model.classes_).index(1)])
    trend_ratio = float(row["MA_20_Ratio"].iloc[0])
    # Signal is based on the model score only. Trend is reported separately,
    # so it cannot silently turn every high-probability stock into AVOID / WAIT.
    signal = "BUY CANDIDATE" if positive_probability >= BUY_THRESHOLD else "AVOID / WAIT"
    trend_status = "Above 20-day average" if trend_ratio > 0 else "At/below 20-day average"
    return {
        "ticker": ticker, "company": company_name,
        "latest_data_date": row["Date"].iloc[0] if "Date" in row else "Unknown",
        "latest_adjusted_close": float(row["Adj Close"].iloc[0]) if "Adj Close" in row else None,
        "positive_probability": positive_probability, "ma_20_ratio": trend_ratio,
        "trend_status": trend_status, "signal": signal, "buy_threshold": BUY_THRESHOLD,
    }

def main() -> None:
    results = []
    stock_universe = load_selected_universe()
    for ticker, company in stock_universe.items():
        try:
            result = predict_stock(ticker, company)
            if result:
                results.append(result)
                print(f"{ticker:16} {result['signal']:14} probability={result['positive_probability']:.1%} "
                      f"trend={result['trend_status']} price={result['latest_adjusted_close']}")
        except Exception as exc:
            print(f"Prediction failed for {ticker}: {exc}")
    if not results:
        raise RuntimeError("No predictions were produced. Train models and check data download errors.")
    output = pd.DataFrame(results).sort_values(["signal", "positive_probability"], ascending=[True, False])
    # Put buy candidates first and rank by probability.
    output["signal_rank"] = output["signal"].map({"BUY CANDIDATE": 1, "AVOID / WAIT": 2})
    output = output.sort_values(["signal_rank", "positive_probability"], ascending=[True, False]).drop(columns="signal_rank")
    output_path = PROCESSED_DATA_DIR / "latest_predictions.csv"
    try:
        output.to_csv(output_path, index=False)
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROCESSED_DATA_DIR / f"latest_predictions_{stamp}.csv"
        output.to_csv(output_path, index=False)
    print("\nFinal ranked predictions:")
    print(output[["ticker", "company", "signal", "positive_probability", "latest_adjusted_close"]].to_string(index=False, formatters={"positive_probability": "{:.1%}".format}))
    print(f"\nSaved: {output_path}")
    print(f"BUY CANDIDATE means model probability >= {BUY_THRESHOLD:.0%}; trend_status is shown separately. This is a screening signal, not a guarantee of profit.")

if __name__ == "__main__":
    main()
