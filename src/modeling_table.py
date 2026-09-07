"""
Assemble the final modeling table: early-window features joined to the
leakage-free target, restricted to eligible loans only.
"""


def assemble_modeling_table(features_early, target, eligible_loan_ids):
    """Inner-join features and target, restricted to eligible loans.

    Returns a single DataFrame with feature columns plus "is_bad".
    """
    features_final = features_early.loc[features_early.index.isin(eligible_loan_ids)]
    target_final = target.loc[target.index.isin(eligible_loan_ids)]
    modeling_table = features_final.join(target_final, how="inner")
    return modeling_table
