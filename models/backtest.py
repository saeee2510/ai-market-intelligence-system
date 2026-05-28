import numpy as np
import pandas as pd


def backtest(df):

    df = df.copy()

    # -----------------------------
    # STRATEGY RETURNS
    # -----------------------------
    df["strategy_return"] = df["signal"] * df["future_return"]

    # replace NaNs safely
    df = df.fillna(0)

    # -----------------------------
    # EQUITY CURVE
    # -----------------------------
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()

    total_return = df["equity_curve"].iloc[-1] - 1

    # -----------------------------
    # SHARPE RATIO
    # -----------------------------
    sharpe = (
        df["strategy_return"].mean() /
        (df["strategy_return"].std() + 1e-9)
    ) * np.sqrt(252)

    # -----------------------------
    # MAX DRAWDOWN
    # -----------------------------
    rolling_max = df["equity_curve"].cummax()
    drawdown = df["equity_curve"] - rolling_max
    max_drawdown = drawdown.min()

    # -----------------------------
    # RESULTS
    # -----------------------------
    print("\n📊 BACKTEST RESULTS")
    print(f"Total Return: {total_return:.4f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")

    return df