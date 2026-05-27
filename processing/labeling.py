import pandas as pd


def create_labels(df, forecast_horizon=3):

    df = df.copy()

    # future price movement
    df["future_return"] = (
        df["Close"].shift(-forecast_horizon) - df["Close"]
    ) / df["Close"]

    # binary classification label
    df["label"] = (df["future_return"] > 0).astype(int)

    # remove NaNs caused by shifting
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("Labeling module ready.")