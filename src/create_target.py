from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PREDICTION_HORIZON = 5

def create_target(input_file: Path, output_file: Path) -> None:
    data = pd.read_csv(input_file)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Adj Close"] = pd.to_numeric(data["Adj Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Adj Close"]).sort_values("Date").reset_index(drop=True)
    data["Future_Close"] = data["Adj Close"].shift(-PREDICTION_HORIZON)
    data["Future_Return_5D"] = data["Future_Close"] / data["Adj Close"] - 1
    data["Target"] = (data["Future_Return_5D"] > 0).astype(int)
    data = data.dropna(subset=["Future_Close", "Future_Return_5D"]).reset_index(drop=True)
    data.to_csv(output_file, index=False)
    print(f"{input_file.name}: created {len(data)} labelled rows")

def process_all_stocks() -> None:
    files = sorted(PROCESSED_DATA_DIR.glob("*_features.csv"))
    if not files:
        raise FileNotFoundError(f"No *_features.csv files found in {PROCESSED_DATA_DIR}")
    for input_file in files:
        output_file = PROCESSED_DATA_DIR / f"{input_file.stem.replace('_features', '')}_model_data.csv"
        try:
            create_target(input_file, output_file)
        except Exception as exc:
            print(f"Failed to create target for {input_file.name}: {exc}")

if __name__ == "__main__":
    process_all_stocks()
