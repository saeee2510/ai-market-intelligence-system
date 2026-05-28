import pandas as pd
import numpy as np


def create_labels(df, horizon=3, threshold=0.005):

    df = df.copy()

    # -----------------------------
    # FUTURE RETURN (horizon-based)
    # -----------------------------
    df["future_return"] = (
        df["Close"].shift(-horizon) - df["Close"]
    ) / df["Close"]

    # -----------------------------
    # LABEL CREATION
    # 1 = UP, 0 = DOWN
    # -----------------------------
    df["label"] = np.where(
        df["future_return"] > threshold, 1,
        np.where(df["future_return"] < -threshold, 0, np.nan)
    )

    # -----------------------------
    # CLEAN DATA
    # -----------------------------
    df = df.dropna().reset_index(drop=True)

    return df


# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":

    sample = pd.DataFrame({
        "Close": [100, 102, 101, 105, 110]
    })

    result = create_labels(sample)

    print(result)