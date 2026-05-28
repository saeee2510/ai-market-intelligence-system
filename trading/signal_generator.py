import numpy as np

def generate_signals(model, X_test, y_test, df_test):

    probs = model.predict_proba(X_test)[:, 1]

    results = df_test.copy()
    results["prob_up"] = probs

    # ONLY trade high-confidence signals
    results["signal"] = np.where(
        results["prob_up"] > 0.7, 1,
        np.where(results["prob_up"] < 0.3, -1, 0)
    )

    return results