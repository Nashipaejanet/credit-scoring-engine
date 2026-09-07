"""
Cost-based decision threshold optimization.

Instead of the naive 0.5 cutoff, this searches for the probability
threshold that minimizes expected business cost, given:
  - cost of a false negative (approving a loan that later goes bad)
    ~= average loan price (the loss on a full default)
  - cost of a false positive (rejecting a loan that would've been fine)
    ~= average financing markup (the profit foregone)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config import THRESHOLD_MAX, THRESHOLD_MIN, THRESHOLD_STEPS


def estimate_costs(demo):
    """Rough per-decision cost estimates derived from the demo table.

    cost_fn: average loan price (full default loss)
    cost_fp: average markup, i.e. loan price minus cash price (foregone profit)
    """
    avg_loan_price = demo["LOAN_PRICE"].mean()
    avg_cash_price = demo["CASH_PRICE"].mean()
    avg_markup = avg_loan_price - avg_cash_price
    return {"cost_fn": avg_loan_price, "cost_fp": avg_markup}


def sweep_thresholds(y_test, proba, cost_fn, cost_fp, t_min=THRESHOLD_MIN, t_max=THRESHOLD_MAX, steps=THRESHOLD_STEPS):
    """Evaluate expected cost across a grid of thresholds and return the
    results sorted best (lowest expected cost) first."""
    thresholds = np.linspace(t_min, t_max, steps)
    results = []

    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        expected_cost = fn * cost_fn + fp * cost_fp
        results.append({"threshold": t, "expected_cost": expected_cost, "fn": fn, "fp": fp})

    return pd.DataFrame(results).sort_values("expected_cost").reset_index(drop=True)


def compare_to_default(threshold_results, default_threshold=0.5):
    """Compare the cost-optimal threshold to the naive default and report
    the percentage cost reduction."""
    default_row = threshold_results[threshold_results["threshold"].round(2) == round(default_threshold, 2)].iloc[0]
    optimal_row = threshold_results.iloc[0]

    cost_reduction_pct = (
        (default_row["expected_cost"] - optimal_row["expected_cost"]) / default_row["expected_cost"] * 100
    )

    return {
        "default_threshold": default_row["threshold"],
        "default_expected_cost": default_row["expected_cost"],
        "optimal_threshold": optimal_row["threshold"],
        "optimal_expected_cost": optimal_row["expected_cost"],
        "cost_reduction_pct": cost_reduction_pct,
    }
