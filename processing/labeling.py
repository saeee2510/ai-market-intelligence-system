import numpy as np
import pandas as pd


def create_labels(df):

    df = df.copy()

    df["future_return"] = df["Close"].shift(-5).pct_change()

    threshold = df["future_return"].abs().quantile(0.75)

    df["label"] = np.where(
        df["future_return"] > threshold, 1,
        np.where(df["future_return"] < -threshold, 0, np.nan)
    )

    df = df.dropna().reset_index(drop=True)

    return df