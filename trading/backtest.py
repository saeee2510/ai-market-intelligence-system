import numpy as np

def backtest(df):

    df = df.copy()

    df["strategy_return"] = df["signal"] * df["future_return"]

    cumulative_return = (1 + df["strategy_return"]).cumprod()[-1] - 1

    sharpe = (
        df["strategy_return"].mean() /
        (df["strategy_return"].std() + 1e-9)
    ) * np.sqrt(252)

    max_drawdown = (df["strategy_return"].cumsum().cummax() -
                    df["strategy_return"].cumsum()).max()

    print("\n📊 BACKTEST RESULTS")
    print(f"Total Return: {cumulative_return:.4f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")