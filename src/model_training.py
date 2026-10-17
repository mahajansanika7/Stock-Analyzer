from pathlib import Path
import warnings
import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from backtesting import backtest_stock, calculate_metrics
from model_utils import create_model

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_HORIZON = 5
N_SPLITS = 5
FEATURE_COLUMNS = ["Daily_Return", "Return_5D", "Return_20D", "MA_20_Ratio", "MA_50_Ratio", "Volatility_20D", "RSI_14"]
TARGET_COLUMN = "Target"

def load_stock_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN, "Adj Close", "Future_Return_5D", "Date"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in FEATURE_COLUMNS + [TARGET_COLUMN, "Adj Close", "Future_Return_5D"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=required).sort_values("Date").reset_index(drop=True)
    if len(data) < 150:
        raise ValueError(f"Only {len(data)} labelled rows; at least 150 are required")
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Target contains only one class")
    return data

def train_and_evaluate(data: pd.DataFrame):
    data = data.sort_values("Date").reset_index(drop=True)
    X, y = data[FEATURE_COLUMNS], data[TARGET_COLUMN].astype(int)
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, gap=PREDICTION_HORIZON)
    predictions = pd.Series(float("nan"), index=data.index, dtype="float64")
    probabilities = pd.Series(float("nan"), index=data.index, dtype="float64")
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = create_model()
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        predictions.iloc[test_idx] = pred
        probabilities.iloc[test_idx] = prob
        try:
            auc = roc_auc_score(y_test, prob) if y_test.nunique() > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        result = {"fold": fold, "train_rows": len(train_idx), "test_rows": len(test_idx),
                  "accuracy": accuracy_score(y_test, pred),
                  "precision": precision_score(y_test, pred, zero_division=0),
                  "recall": recall_score(y_test, pred, zero_division=0),
                  "f1": f1_score(y_test, pred, zero_division=0), "roc_auc": auc}
        fold_results.append(result)
        print(f"Fold {fold}: accuracy={result['accuracy']:.3f}, precision={result['precision']:.3f}, recall={result['recall']:.3f}, AUC={result['roc_auc']:.3f}")
        print("  Confusion matrix [[TN, FP], [FN, TP]]:")
        print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print("\nOut-of-sample backtest (historical; not a guarantee of future results):")
    backtest = backtest_stock(data, predictions, probabilities, holding_period=PREDICTION_HORIZON)
    metrics = calculate_metrics(backtest)
    print(f"  Strategy return : {metrics['strategy_return']:.2%}")
    print(f"  Buy & hold return: {metrics['buy_hold_return']:.2%}")
    print(f"  Max drawdown    : {metrics['max_drawdown']:.2%}")
    print(f"  Trades / win rate: {metrics['number_of_trades']} / {metrics['win_rate']:.2%}")
    # Deployment model uses all labelled history; latest five rows are unlabelled and excluded by target creation.
    final_model = create_model()
    final_model.fit(X, y)
    return final_model, fold_results, backtest

def main() -> None:
    files = sorted(PROCESSED_DATA_DIR.glob("*_model_data.csv"))
    if not files:
        raise FileNotFoundError("No model data. Run the full pipeline first.")
    summary = []
    for file in files:
        print("\n" + "=" * 65 + f"\nTraining {file.name}\n" + "=" * 65)
        try:
            data = load_stock_data(file)
            model, folds, backtest = train_and_evaluate(data)
            stock = file.stem.replace("_model_data", "")
            model_path = MODEL_DIR / f"{stock}_final_model.joblib"
            joblib.dump(model, model_path)
            pd.DataFrame(folds).to_csv(PROCESSED_DATA_DIR / f"{stock}_training_metrics.csv", index=False)
            backtest.to_csv(PROCESSED_DATA_DIR / f"{stock}_backtest.csv", index=False)
            summary.append({"ticker": stock, "rows": len(data), "mean_cv_accuracy": pd.DataFrame(folds)["accuracy"].mean(),
                            "mean_cv_auc": pd.DataFrame(folds)["roc_auc"].mean(), "model_type": "RandomForestClassifier", "model_file": model_path.name})
            print(f"Saved final model: {model_path}")
        except Exception as exc:
            warnings.warn(f"Skipping {file.name}: {exc}")
    if not summary:
        raise RuntimeError("No models were trained successfully. Check the errors above.")
    pd.DataFrame(summary).to_csv(PROCESSED_DATA_DIR / "model_summary.csv", index=False)
    print(f"\nTrained {len(summary)} models. Summary: {PROCESSED_DATA_DIR / 'model_summary.csv'}")

if __name__ == "__main__":
    main()
