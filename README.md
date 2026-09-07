# Credit Scoring Engine

A leakage-free, fairness-audited credit scoring model built on MoPhones
loan portfolio data, using an out-of-time train/test design so the model
is never scored on information it wouldn't have had at decision time.

## What it does

1. **Loads** five quarterly credit snapshots plus a sales/demographics
   workbook and an NPS survey.
2. **Splits** the credit history at a cutoff date (31 Mar 2025) into an
   early window (used for features) and a late window (used to define the
   outcome) — an out-of-time design that prevents the model from seeing
   the future.
3. **Engineers features** from the early window only: worst/average days
   past due, times in arrears, a data-quality flag for ARREARS/DPD
   mismatches (never imputed — flagged), collection rate, months on book,
   and utilisation ratio.
4. **Builds a leakage-free target**: excludes loans already in a bad or
   ambiguous state at the cutoff, then labels the rest "bad" if they reach
   FPD/FMD/PAR 30 status in the late window.
5. **Trains** a scaled Logistic Regression classifier and reports AUC.
6. **Audits fairness** using the four-fifths (disparate impact) rule
   across gender and age band.
7. **Optimizes the decision threshold** by expected business cost (cost of
   a missed default vs. cost of rejecting a good loan), instead of using
   the naive 0.5 cutoff.

## Project structure

```
credit_scoring_engine/
├── config.py           # paths, cutoff dates, feature list, thresholds
├── run.py               # entry point — runs the full pipeline
├── requirements.txt
└── src/
    ├── data_loader.py    # load + clean credit / demo / NPS sources
    ├── features.py       # early-window feature engineering
    ├── target.py         # leakage-free target + eligibility filtering
    ├── modeling_table.py # joins features + target
    ├── model.py          # train/test split, scaling, LR training, eval
    ├── fairness.py        # disparate impact / four-fifths audit
    ├── threshold.py       # cost-based threshold search
    └── pipeline.py         # orchestrates all of the above
```

## Setup

```bash
pip install -r requirements.txt
```

Set the `CREDIT_DATA_DIR` environment variable to point at your local folder
containing the raw data (or edit the fallback path in `config.py` directly):

- `Credit Data - 01-01-2025.csv` … `Credit Data - 30-12-2025.csv` (5 files)
- `Sales and Customer Data.xlsx`
- `NPS Data.xlsx`

On Windows (Command Prompt):
```
set CREDIT_DATA_DIR=C:\path\to\your\data\folder
```

This data is not included in the repo — see `.gitignore`.

## Run

```bash
python run.py
```

This prints load/feature/model progress, the AUC, the fairness audit
result per protected attribute, and the top cost-optimal thresholds.

## Findings

Verified results from a full run on the real MoPhones portfolio data:

| Metric | Result |
|---|---|
| Eligible loans (leakage-free) | 6,412 |
| Bad rate | 14.5% |
| Model AUC | 0.917 |
| Cost-optimal threshold | 0.06 (vs. naive 0.50) |
| Cost reduction vs. naive threshold | 46.8% |

**Fairness audit:**
- **Gender** — passes the four-fifths rule. Male approval rate 70.1%, female 74.3% (di_ratio 0.94).
- **Age band** — the **18-25 group fails** the four-fifths rule, with a di_ratio of 0.63 (approval rate 53.6% vs. 85.7% for the 56+ reference group). All other age bands (26-35 through 56+) pass.

This is a genuine finding, not a modeling artifact — it's flagged here rather than smoothed over. Younger borrowers are being approved at a meaningfully lower rate than older ones, which is the kind of disparate impact this audit exists to catch. Next steps worth investigating: whether age-correlated features (e.g. months on book, utilisation ratio) are acting as a proxy for age, and whether a fairness constraint should be added to model training rather than relying on threshold adjustment alone.

## Design notes

- **No DPD imputation from ARREARS.** Where `ARREARS > 0` but
  `DAYS_PAST_DUE == 0`, that's flagged as a feature
  (`arrears_dpd_mismatch_ever`) rather than "corrected" — inventing a DPD
  value from ARREARS would be fabricating data the source system doesn't
  actually provide.
- **Out-of-time, not random, split.** Features come strictly from before
  the cutoff; the label comes strictly from after it. A random train/test
  split on this kind of panel data would leak future information into
  the features.
- **Cost-based threshold, not accuracy-based.** A missed default (false
  negative) and a wrongly rejected good loan (false positive) have very
  different dollar costs here, so the threshold is chosen to minimize
  expected cost, not to maximize accuracy or F1.
