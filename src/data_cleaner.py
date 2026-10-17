from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def clean_stock_file(input_file, output_file):
    data = pd.read_csv(input_file)
    initial_rows = len(data)

    # Convert Date column
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    # Remove rows with invalid dates
    data = data.dropna(subset=["Date"])

    # Remove duplicate dates
    data = data.drop_duplicates(subset=["Date"])

    # Required price columns
    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close"
    ]

    # Remove rows with missing prices
    data = data.dropna(subset=price_columns)

    # Prices must be positive
    valid_prices = (
        (data["Open"] > 0)
        & (data["High"] > 0)
        & (data["Low"] > 0)
        & (data["Close"] > 0)
        & (data["Adj Close"] > 0)
    )

    data = data[valid_prices]

    # High must be >= Low
    data = data[data["High"] >= data["Low"]]

    # Open and Close must lie between Low and High
    valid_ohlc = (
        (data["Open"] >= data["Low"])
        & (data["Open"] <= data["High"])
        & (data["Close"] >= data["Low"])
        & (data["Close"] <= data["High"])
    )

    data = data[valid_ohlc]

    # Volume cannot be negative when the provider supplies it.
    if "Volume" in data.columns:
        data = data[data["Volume"].fillna(0) >= 0]

    # Sort chronologically
    data = data.sort_values("Date")

    # Reset pandas index
    data = data.reset_index(drop=True)

    # Save cleaned data
    data.to_csv(output_file, index=False)

    removed_rows = initial_rows - len(data)

    print(
        f"{input_file.name}: "
        f"{initial_rows} → {len(data)} rows "
        f"({removed_rows} removed)"
    )


def clean_all_stocks():

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stock_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not stock_files:
        print("No stock data found in data/raw/")
        return

    print(f"Found {len(stock_files)} stock files.\n")

    for input_file in stock_files:

        output_file = (
            PROCESSED_DATA_DIR
            / f"{input_file.stem}_cleaned.csv"
        )

        try:
            clean_stock_file(
                input_file,
                output_file
            )

        except Exception as e:
            print(
                f"Failed to clean "
                f"{input_file.name}: {e}"
            )


if __name__ == "__main__":
    clean_all_stocks()