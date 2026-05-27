import pandas as pd


def create_labels(df, horizon=3):

    df = df.copy()

    # future percentage return
    df["future_return"] = (
        df["Close"].shift(-horizon) / df["Close"]
    ) - 1

    # binary classification label
    # 1 = stock goes up
    # 0 = stock goes down
    df["label"] = (df["future_return"] > 0).astype(int)

    # remove last rows with NaN future values
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":

    sample = pd.DataFrame({
        "Close": [100, 102, 101, 105, 110]
    })

    result = create_labels(sample)

    print(result)