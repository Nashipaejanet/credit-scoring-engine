"""
Train and evaluate the credit-risk classifier.

Kept deliberately simple (scaled Logistic Regression) so coefficients stay
interpretable for the fairness and threshold-optimization steps downstream.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE


def prepare_xy(modeling_table, feature_cols=None):
    """Split the modeling table into X (features) and y (is_bad).

    Casts the boolean mismatch flag to int and fills missing values with
    the column median.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    X = modeling_table[feature_cols].copy()
    if "arrears_dpd_mismatch_ever" in X.columns:
        X["arrears_dpd_mismatch_ever"] = X["arrears_dpd_mismatch_ever"].astype(int)
    X = X.fillna(X.median())
    y = modeling_table["is_bad"]
    return X, y


def train_test_split_scaled(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified train/test split, with features standardized on the
    training set and that same scaling applied to the test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    """Fit a Logistic Regression classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return predicted probabilities and AUC on the held-out test set."""
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    return proba, auc


def coefficient_summary(model, feature_cols):
    """Feature coefficients sorted by absolute magnitude (scaled model
    coefficients are directly comparable to each other)."""
    return pd.Series(model.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
