import pandas as pd

TRANSACTION_COST = 0.001  # 0.1% per side: entry and exit


def backtest_stock(
    data: pd.DataFrame,
    predictions,
    probabilities=None,
    holding_period=5,
    transaction_cost=TRANSACTION_COST,
):
    if data.empty:
        raise ValueError("No data provided.")

    if len(data) != len(predictions):
        raise ValueError("Data and predictions must have the same number of rows.")

    if holding_period < 1:
        raise ValueError("Holding period must be at least 1.")

    if not 0 <= transaction_cost < 1:
        raise ValueError("Transaction cost must be between 0 and 1.")

    if "Adj Close" not in data.columns:
        raise ValueError("Missing required column: Adj Close")

    result = data.copy().reset_index(drop=True)
    result["Prediction"] = list(predictions)

    if probabilities is not None:
        if len(probabilities) != len(result):
            raise ValueError(
                "Data and probabilities must have the same number of rows."
            )
        result["Probability"] = list(probabilities)

    result["Adj Close"] = pd.to_numeric(result["Adj Close"], errors="coerce")

    if result["Adj Close"].isna().any() or (result["Adj Close"] <= 0).any():
        raise ValueError("Adjusted closing prices must be valid and positive.")

    # A signal on day t is evaluated against the close holding_period
    # trading rows later.
    result["Exit_Price"] = result["Adj Close"].shift(-holding_period)
    result["Actual_Return"] = result["Exit_Price"] / result["Adj Close"] - 1

    result["Position"] = 0
    result["Strategy_Return"] = 0.0
    result["Trade_Return"] = float("nan")

    # Daily market returns, used only while the strategy holds a position.
    daily_returns = result["Adj Close"].pct_change().fillna(0.0)

    next_available_entry = 0
    trade_count = 0

    for i in range(len(result)):
        prediction = result.loc[i, "Prediction"]

        # NaN predictions represent training rows or rows outside test folds.
        if pd.isna(prediction):
            continue

        # Do not enter another trade while a previous trade is open.
        if i < next_available_entry:
            continue

        # Ensure this trade has a known exit price.
        if i + holding_period >= len(result):
            continue

        if int(prediction) != 1:
            continue

        exit_idx = i + holding_period
        gross_return = float(result.loc[i, "Actual_Return"])

        result.loc[i, "Position"] = 1
        result.loc[i, "Trade_Return"] = gross_return
        trade_count += 1

        # Apply entry transaction cost on the entry row.
        result.loc[i, "Strategy_Return"] = (1 - transaction_cost) - 1

        # Mark the position to market each day until exit.
        for j in range(i + 1, exit_idx + 1):
            result.loc[j, "Strategy_Return"] = daily_returns.iloc[j]

        # Apply exit transaction cost on the exit row.
        result.loc[exit_idx, "Strategy_Return"] = (
            1 + result.loc[exit_idx, "Strategy_Return"]
        ) * (1 - transaction_cost) - 1

        # Prevent overlapping trades.
        next_available_entry = exit_idx + 1

    # Evaluate strategy and benchmark over the same continuous date range.
    valid_prediction_rows = result.index[result["Prediction"].notna()]

    if len(valid_prediction_rows) == 0:
        raise ValueError("No out-of-sample predictions available.")

    first_idx = int(valid_prediction_rows[0])
    last_idx = min(
        int(valid_prediction_rows[-1]) + holding_period,
        len(result) - 1,
    )

    result = result.loc[first_idx:last_idx].copy().reset_index(drop=True)

    # Equity curves start at 1.0 and cover the same dates.
    result["Strategy_Equity"] = (1 + result["Strategy_Return"]).cumprod()

    first_price = result["Adj Close"].iloc[0]
    result["Buy_Hold_Equity"] = result["Adj Close"] / first_price

    result["Running_Peak"] = result["Strategy_Equity"].cummax()
    result["Drawdown"] = (result["Strategy_Equity"] / result["Running_Peak"]) - 1

    return result


def calculate_metrics(result):
    if result.empty:
        raise ValueError("No results available.")

    strategy_return = result["Strategy_Equity"].iloc[-1] - 1
    buy_hold_return = result["Buy_Hold_Equity"].iloc[-1] - 1
    max_drawdown = result["Drawdown"].min()

    # Count actual trade entries, not days spent holding a position.
    trades = result[result["Position"] == 1]

    win_rate = (trades["Trade_Return"] > 0).mean() if not trades.empty else 0.0

    return {
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "max_drawdown": max_drawdown,
        "number_of_trades": len(trades),
        "win_rate": win_rate,
    }
