from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(period).mean()
    average_loss = loss.rolling(period).mean()

    relative_strength = average_gain / average_loss.replace(0, float("nan"))

    rsi = 100 - (100 / (1 + relative_strength))
    # No losses over the lookback window means RSI is 100; no gains/losses means 50.
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)

    return rsi


def create_features(input_file, output_file):

    data = pd.read_csv(input_file)

    # -----------------------------
    # Return features
    # -----------------------------

    data["Daily_Return"] = data["Adj Close"].pct_change()

    data["Return_5D"] = data["Adj Close"].pct_change(periods=5)

    data["Return_20D"] = data["Adj Close"].pct_change(periods=20)

    # -----------------------------
    # Moving averages
    # -----------------------------

    ma_20 = data["Adj Close"].rolling(20).mean()
    ma_50 = data["Adj Close"].rolling(50).mean()

    data["MA_20_Ratio"] = data["Adj Close"] / ma_20 - 1
    data["MA_50_Ratio"] = data["Adj Close"] / ma_50 - 1

    # -----------------------------
    # Volatility
    # -----------------------------

    data["Volatility_20D"] = data["Daily_Return"].rolling(20).std()

    # -----------------------------
    # RSI
    # -----------------------------

    data["RSI_14"] = calculate_rsi(data["Adj Close"], period=14)

    # Remove rows where indicators
    # cannot yet be calculated
    data = data.dropna().reset_index(drop=True)

    data.to_csv(output_file, index=False)

    print(f"{input_file.name}: " f"features created → {output_file.name}")


def process_all_stocks():

    feature_files = list(PROCESSED_DATA_DIR.glob("*_cleaned.csv"))

    if not feature_files:
        print("No cleaned stock files found.")
        return

    for input_file in feature_files:

        output_file = (
            PROCESSED_DATA_DIR
            / f"{input_file.stem.replace('_cleaned', '')}_features.csv"
        )

        try:
            create_features(input_file, output_file)

        except Exception as e:
            print(f"Failed for {input_file.name}: {e}")


if __name__ == "__main__":
    process_all_stocks()
