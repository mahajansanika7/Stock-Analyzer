"""Model factory for StockAnalyzer.

Random Forest is used for tabular technical-indicator features. The model is
kept intentionally regularized because daily stock data is noisy and small.
"""
from sklearn.ensemble import RandomForestClassifier


def create_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=5,
        min_samples_split=12,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
