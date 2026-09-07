"""
Load and lightly clean the three raw MoPhones sources:
  1. Credit snapshots (5 CSVs, one per quarter) -> credit_panel
  2. Sales & customer workbook (sales / gender / DOB / income tabs) -> demo
  3. NPS survey workbook -> nps_features

Each loader returns a clean, deduplicated DataFrame indexed by "Loan Id"
(or containing it as a column) so downstream modules can merge on it
without re-doing any of this cleanup.
"""

import os

import pandas as pd

from config import (
    DATA_DIR,
    CREDIT_SNAPSHOT_FILES,
    SALES_CUSTOMER_WORKBOOK,
    NPS_WORKBOOK,
)


def load_credit_panel(data_dir=DATA_DIR, files=None):
    """Concatenate the quarterly credit snapshots into one long panel.

    Each row is one loan's state at one snapshot date, so a loan observed
    across all 5 snapshots contributes up to 5 rows.
    """
    files = files or CREDIT_SNAPSHOT_FILES
    frames = []
    for filename in files:
        path = os.path.join(data_dir, filename)
        df = pd.read_csv(path)
        df["snapshot_date"] = pd.to_datetime(df["DATE"])
        frames.append(df)

    credit_panel = pd.concat(frames, ignore_index=True)
    return credit_panel


def load_demographics(data_dir=DATA_DIR, workbook=SALES_CUSTOMER_WORKBOOK):
    """Build one row per loan with sale terms, gender, age and income.

    Pulls from four tabs of the same workbook (Sales Details, Gender, DOB,
    Income Level) and merges them on "Loan Id".
    """
    xl = pd.ExcelFile(os.path.join(data_dir, workbook))

    sales = xl.parse("Sales Details").dropna(how="all")
    gender = xl.parse("Gender").dropna(how="all")
    dob = xl.parse("DOB").dropna(how="all")
    income_raw = xl.parse("Income Level").dropna(how="all")

    # --- Sales: one row per loan ---
    sales = sales.drop_duplicates("Loan Id")

    # --- Gender: normalise M/F, fill blanks ---
    gender["Gender"] = gender["Gender"].replace({"M": "Male", "F": "Female"}).fillna("Unspecified")
    gender = gender.drop_duplicates("Loan Id")

    # --- DOB: column has a trailing space ("Loan Id "); keep latest record ---
    dob["createdAt UTC"] = pd.to_datetime(dob["createdAt UTC"], utc=True)
    dob = dob.sort_values("createdAt UTC").drop_duplicates("Loan Id ", keep="last")

    # --- Income: derive a monthly figure from Received / Duration ---
    income_raw = income_raw.dropna(subset=["Loan Id"])
    income_raw["monthly_income"] = income_raw["Received"] / income_raw["Duration"]
    income = income_raw.groupby("Loan Id", as_index=False)[["monthly_income"]].mean()

    # --- Merge into a single demo table ---
    demo = sales[["Loan Id", "SALE_DATE", "SALE_TYPE", "LOAN_TERM", "CASH_PRICE", "LOAN_PRICE"]].drop_duplicates("Loan Id")
    demo = demo.merge(gender[["Loan Id", "Gender", "Citizenship"]], on="Loan Id", how="left")
    demo = demo.merge(dob[["Loan Id ", "date_of_birth"]], left_on="Loan Id", right_on="Loan Id ", how="left")
    demo = demo.drop(columns=["Loan Id "], errors="ignore")
    demo = demo.merge(income, on="Loan Id", how="left")
    demo = demo.rename(columns={"monthly_income": "avg_monthly_income"})
    demo["avg_monthly_income"] = demo["avg_monthly_income"].clip(upper=1_000_000)

    demo = _add_age(demo)
    return demo


def _add_age(demo):
    """Derive age in years and an age band, dropping implausible ages (>75)."""
    demo = demo.copy()
    demo["date_of_birth"] = pd.to_datetime(demo["date_of_birth"], utc=True)
    reference_date = pd.Timestamp.now(tz="UTC")
    demo["age_years"] = (reference_date - demo["date_of_birth"]).dt.days / 365.25

    implausible = demo["age_years"] > 75
    demo.loc[implausible, "age_years"] = None

    demo["age_band"] = pd.cut(
        demo["age_years"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    )
    return demo


def load_nps(data_dir=DATA_DIR, workbook=NPS_WORKBOOK):
    """Load the NPS survey and rename the long question columns.

    Keeps one response per loan (the most recent) and returns three
    renamed fields: nps_score, phone_locked_despite_payment,
    payment_delay_experienced.
    """
    nps = pd.read_excel(os.path.join(data_dir, workbook)).dropna(how="all")
    nps["Submitted at"] = pd.to_datetime(nps["Submitted at"])
    nps = nps.sort_values("Submitted at").drop_duplicates("Loan Id", keep="last")

    score_col = (
        "Using a scale from 0 (not likely) to 10 (very likely), how likely "
        "are you to recommend MoPhones to friends or family?"
    )
    lock_col = "Have you ever had your phone lock despite making a payment on time?"
    delay_col = "Have you ever experienced a delay in your payment reflecting in your Mophones account?"

    nps_features = nps[["Loan Id", score_col, lock_col, delay_col]].copy()
    nps_features = nps_features.rename(
        columns={
            score_col: "nps_score",
            lock_col: "phone_locked_despite_payment",
            delay_col: "payment_delay_experienced",
        }
    )
    nps_features["phone_locked_despite_payment"] = nps_features["phone_locked_despite_payment"] == "Yes"
    nps_features["payment_delay_experienced"] = nps_features["payment_delay_experienced"] == "Yes"

    return nps_features
