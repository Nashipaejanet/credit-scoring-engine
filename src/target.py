"""
Build the leakage-free modeling target.

A loan is only eligible for training if:
  (a) it existed early enough to have early-window features, AND
  (b) it wasn't already in a bad/excluded state as of the cutoff
      (otherwise the "prediction" would just be restating the past).

The label itself, is_bad, is whether the loan reaches a bad status
(FPD / FMD / PAR 30) at any point in the late window.
"""

from config import BAD_STATUSES, EXCLUDE_AT_CUTOFF_STATUSES


def status_at_cutoff(early_panel):
    """Each loan's most recent status as of the early-window cutoff."""
    return early_panel.sort_values("snapshot_date").groupby("LOAN_ID")["ACCOUNT_STATUS_L2"].last()


def status_in_late_window(late_panel):
    """Each loan's most recent status observed in the late (outcome) window."""
    return late_panel.sort_values("snapshot_date").groupby("LOAN_ID")["ACCOUNT_STATUS_L2"].last()


def eligible_loans(early_panel, cutoff_status, exclude_statuses=None):
    """Loans that existed by the cutoff and weren't already bad/excluded then."""
    exclude_statuses = exclude_statuses or EXCLUDE_AT_CUTOFF_STATUSES
    excluded = cutoff_status.isin(exclude_statuses)
    all_early_loans = early_panel["LOAN_ID"].unique()
    return [loan for loan in all_early_loans if loan not in excluded[excluded].index]


def build_target(late_status, bad_statuses=None):
    """1 if the loan reaches a bad status in the late window, else 0."""
    bad_statuses = bad_statuses or BAD_STATUSES
    return late_status.isin(bad_statuses).astype(int).rename("is_bad")
