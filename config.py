"""
Central configuration for the Credit Scoring Engine.

Edit DATA_DIR to point at the folder containing the raw MoPhones files:
    Credit Data - 01-01-2025.csv
    Credit Data - 30-03-2025.csv
    Credit Data - 30-06-2025.csv
    Credit Data - 30-09-2025.csv
    Credit Data - 30-12-2025.csv
    Sales and Customer Data.xlsx
    NPS Data.xlsx
"""

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Set the CREDIT_DATA_DIR environment variable to point at your local data
# folder, or edit the fallback path below directly.
DATA_DIR = Path(os.environ.get("CREDIT_DATA_DIR", "./data"))

CREDIT_SNAPSHOT_FILES = [
    "Credit Data - 01-01-2025.csv",
    "Credit Data - 30-03-2025.csv",
    "Credit Data - 30-06-2025.csv",
    "Credit Data - 30-09-2025.csv",
    "Credit Data - 30-12-2025.csv",
]

SALES_CUSTOMER_WORKBOOK = "Sales and Customer Data.xlsx"
NPS_WORKBOOK = "NPS Data.xlsx"

# --- Out-of-time split -------------------------------------------------------
# Loans are scored on data up to and including EARLY_CUTOFF ("early panel").
# Whether the loan later goes bad is judged using snapshots after that date
# ("late panel"). This keeps the model from ever seeing the future.
EARLY_CUTOFF = "2025-03-31"

# Statuses that mean a loan was already in trouble by the cutoff. These loans
# are dropped before training so the model isn't scored on cases where the
# outcome was already decided.
EXCLUDE_AT_CUTOFF_STATUSES = ["FPD", "FMD", "PAR 30", "Return", "Inactive", "Unknown"]

# Statuses that define a "bad" loan in the outcome window.
BAD_STATUSES = ["FPD", "FMD", "PAR 30"]

# --- Feature set -------------------------------------------------------------
FEATURE_COLUMNS = [
    "max_dpd",
    "times_in_arrears",
    "arrears_dpd_mismatch_ever",
    "avg_collection_rate",
    "months_on_book",
    "utilisation_ratio",
]

# --- Modeling ----------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Fairness ------------------------------------------------------------
PROTECTED_ATTRIBUTES = ["Gender", "age_band"]
FOUR_FIFTHS_THRESHOLD = 0.80

# --- Cost-based threshold search --------------------------------------------
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
THRESHOLD_STEPS = 99
