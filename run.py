"""
Entry point for the Credit Scoring Engine.

Usage:
    python run.py

Before running, set DATA_DIR in config.py to point at the folder holding
the raw MoPhones files.
"""

from src.pipeline import run_pipeline

if __name__ == "__main__":
    results = run_pipeline(verbose=True)

    print("\n--- Fairness audit detail ---")
    for col, result in results["fairness_results"].items():
        print(f"\nBy {col}:")
        print(result.to_string(index=False))

    print("\n--- Top cost-optimal thresholds ---")
    print(results["threshold_results"].head(5).to_string(index=False))
