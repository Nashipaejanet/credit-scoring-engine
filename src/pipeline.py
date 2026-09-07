"""
End-to-end orchestration: raw files -> cleaned data -> features -> target
-> trained model -> fairness audit -> cost-optimal threshold.

Run via run.py at the project root, or import run_pipeline() directly.
"""

from config import EARLY_CUTOFF, FEATURE_COLUMNS

from src import data_loader, features, target, modeling_table, model, fairness, threshold


def run_pipeline(verbose=True):
    def log(msg):
        if verbose:
            print(msg)

    # 1. Load ------------------------------------------------------------
    log("Loading credit snapshots...")
    credit_panel = data_loader.load_credit_panel()

    log("Loading demographics...")
    demo = data_loader.load_demographics()

    log("Loading NPS survey...")
    nps_features = data_loader.load_nps()

    # 2. Split into early (feature) / late (outcome) windows -------------
    log(f"Splitting panel at {EARLY_CUTOFF}...")
    early_panel, late_panel = features.split_panels(credit_panel)

    # 3. Features ----------------------------------------------------------
    log("Building early-window features...")
    features_early = features.build_early_features(early_panel, demo)

    # 4. Target + eligibility ---------------------------------------------
    log("Building leakage-free target...")
    cutoff_status = target.status_at_cutoff(early_panel)
    late_status = target.status_in_late_window(late_panel)
    is_bad = target.build_target(late_status)
    loans_eligible = target.eligible_loans(early_panel, cutoff_status)

    # 5. Modeling table -----------------------------------------------------
    log("Assembling modeling table...")
    table = modeling_table.assemble_modeling_table(features_early, is_bad, loans_eligible)
    log(f"  {table.shape[0]} eligible loans, bad rate = {table['is_bad'].mean():.1%}")

    # 6. Train / evaluate ----------------------------------------------------
    log("Training model...")
    X, y = model.prepare_xy(table, FEATURE_COLUMNS)
    X_train, X_test, y_train, y_test, scaler = model.train_test_split_scaled(X, y)
    clf = model.train_model(X_train, y_train)
    proba, auc = model.evaluate_model(clf, X_test, y_test)
    log(f"  AUC: {auc:.4f}")
    coefs = model.coefficient_summary(clf, FEATURE_COLUMNS)

    # 7. Fairness audit -------------------------------------------------
    log("Running fairness audit...")
    audit_df = fairness.build_audit_frame(demo, X_test, proba)
    fairness_results = fairness.run_fairness_audit(audit_df)
    for col, result in fairness_results.items():
        log(f"  {col}: passes four-fifths = {result['passes_four_fifths'].all()}")

    # 8. Cost-based threshold optimization --------------------------------
    log("Optimizing decision threshold...")
    costs = threshold.estimate_costs(demo)
    threshold_results = threshold.sweep_thresholds(y_test, proba, costs["cost_fn"], costs["cost_fp"])
    comparison = threshold.compare_to_default(threshold_results)
    log(
        f"  Optimal threshold: {comparison['optimal_threshold']:.2f} "
        f"(cost reduction vs 0.5 default: {comparison['cost_reduction_pct']:.1f}%)"
    )

    return {
        "credit_panel": credit_panel,
        "demo": demo,
        "nps_features": nps_features,
        "modeling_table": table,
        "model": clf,
        "auc": auc,
        "coefficients": coefs,
        "X_test": X_test,
        "y_test": y_test,
        "proba": proba,
        "fairness_results": fairness_results,
        "threshold_results": threshold_results,
        "threshold_comparison": comparison,
    }
