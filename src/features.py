"""
Build out-of-time features from the "early panel" — the credit snapshots
observed on or before EARLY_CUTOFF. Nothing here is allowed to see what
happens after the cutoff; that boundary is what keeps the model honest.
"""

import pandas as pd

from config import EARLY_CUTOFF


def split_panels(credit_panel, early_cutoff=EARLY_CUTOFF):
    """Split the full credit panel into an early (feature) window and a
    late (outcome) window at early_cutoff."""
    cutoff = pd.Timestamp(early_cutoff)
    early_panel = credit_panel[credit_panel["snapshot_date"] <= cutoff].copy()
    late_panel = credit_panel[credit_panel["snapshot_date"] > cutoff].copy()
    return early_panel, late_panel


def build_early_features(early_panel, demo):
    """Compute one feature row per loan from the early panel.

    Features:
      max_dpd                    - worst days-past-due seen in the window
      avg_dpd                    - average days-past-due across snapshots
      times_in_arrears           - count of snapshots flagged "Arrears"
      arrears_dpd_mismatch_ever  - ARREARS > 0 but DPD == 0 at least once
                                    (a data-quality flag, not imputed DPD)
      avg_collection_rate        - total paid / expected paid to date
      months_on_book             - customer age in days / 30, at latest snapshot
      utilisation_ratio          - closing balance / original loan price
    """
    grouped = early_panel.groupby("LOAN_ID")

    max_dpd = grouped["DAYS_PAST_DUE"].max().rename("max_dpd")
    avg_dpd = grouped["DAYS_PAST_DUE"].mean().rename("avg_dpd")
    times_in_arrears = grouped["BALANCE_DUE_STATUS"].apply(
        lambda s: (s == "Arrears").sum()
    ).rename("times_in_arrears")

    # Data-quality flag: ARREARS > 0 while DPD == 0 should not happen.
    # We flag it as a feature rather than imputing a "corrected" DPD —
    # inventing a DPD value from ARREARS would be fabricating data.
    early_panel = early_panel.copy()
    early_panel["mismatch"] = (early_panel["ARREARS"] > 0) & (early_panel["DAYS_PAST_DUE"] == 0)
    mismatch_ever = early_panel.groupby("LOAN_ID")["mismatch"].any().rename("arrears_dpd_mismatch_ever")

    latest_early = early_panel.sort_values("snapshot_date").groupby("LOAN_ID").last()
    latest_early["expected_to_date"] = latest_early["TOTAL_PAID"] - latest_early["BALANCE_DUE_TO_DATE"]
    collection_rate = (latest_early["TOTAL_PAID"] / latest_early["expected_to_date"]).rename("avg_collection_rate")

    months_on_book = (latest_early["CUSTOMER_AGE"] / 30).rename("months_on_book")

    loan_price_early = demo.set_index("Loan Id")["LOAN_PRICE"].reindex(latest_early.index)
    utilisation_ratio = (latest_early["CLOSING_BALANCE"] / loan_price_early).rename("utilisation_ratio")

    features_early = pd.concat(
        [max_dpd, avg_dpd, times_in_arrears, mismatch_ever, collection_rate, months_on_book, utilisation_ratio],
        axis=1,
    )
    return features_early
