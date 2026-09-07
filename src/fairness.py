"""
Fairness audit: checks whether the model's approval decisions differ
disproportionately across protected groups, using the four-fifths (80%)
rule commonly applied in disparate-impact analysis.
"""

import pandas as pd

from config import FOUR_FIFTHS_THRESHOLD


def disparate_impact_ratio(df, decision_col, protected_col, favorable_outcome=0, threshold=None):
    """Compute each group's favorable-outcome rate relative to the group
    with the highest rate, and flag whether it clears the four-fifths rule.

    favorable_outcome=0 assumes decision_col is "predicted_bad" (0/1), so
    the favorable outcome is *not* being flagged as bad, i.e. approval.
    """
    threshold = threshold or FOUR_FIFTHS_THRESHOLD
    rates = df.groupby(protected_col, observed=True)[decision_col].apply(
        lambda x: (x == favorable_outcome).mean()
    )
    reference_rate = rates.max()

    result = pd.DataFrame({"group": rates.index, "approval_rate": rates.values})
    result["di_ratio"] = result["approval_rate"] / reference_rate
    result["passes_four_fifths"] = result["di_ratio"] >= threshold
    return result.sort_values("di_ratio")


def build_audit_frame(demo, X_test, proba, decision_threshold=0.5, protected_cols=("Gender", "age_band")):
    """Join predicted decisions to demographic attributes for the test set.

    decision_threshold controls what counts as a "predicted bad" decision;
    pass the same threshold you intend to actually use in production.
    """
    demo_indexed = demo.set_index("Loan Id")
    test_demographics = demo_indexed.loc[demo_indexed.index.isin(X_test.index), list(protected_cols)]

    decisions = pd.Series(
        (proba >= decision_threshold).astype(int), index=X_test.index, name="predicted_bad"
    )

    audit_df = test_demographics.join(decisions, how="inner").dropna()
    return audit_df


def run_fairness_audit(audit_df, protected_cols=("Gender", "age_band"), threshold=None):
    """Run the disparate-impact check for each protected column and return
    a dict of {column_name: result_dataframe}."""
    return {
        col: disparate_impact_ratio(audit_df, "predicted_bad", col, threshold=threshold)
        for col in protected_cols
    }
