import numpy as np
import pandas as pd


def generate_signals(model, X_test, df_test):

    # -----------------------------
    # GET PROBABILITIES
    # -----------------------------
    probs = model.predict_proba(X_test)[:, 1]

    # IMPORTANT: reset index to avoid mismatch
    df = df_test.copy().reset_index(drop=True)

    # ensure alignment
    df = df.iloc[:len(probs)].copy()

    df["prob_up"] = probs

    # -----------------------------
    # SIGNAL RULES
    # -----------------------------
    df["signal"] = np.where(
        df["prob_up"] > 0.7, 1,
        np.where(df["prob_up"] < 0.3, -1, 0)
    )

    return df