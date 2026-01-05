"""Model building, training and evaluation utilities."""
from typing import Tuple, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def _to_dense(X):
    """Convert sparse matrix to dense numpy array if necessary."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return X
from sklearn.model_selection import train_test_split


def build_pipeline(numeric_cols, categorical_cols, classifier=None) -> Pipeline:
    """Build a preprocessing + classifier pipeline.

    numeric_cols: list of column names with numeric dtype
    categorical_cols: list of column names with object/category dtype
    """
    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    transformers = []
    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_cols))
    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])
        # don't pass 'sparse' or 'sparse_output' to support multiple sklearn versions
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers)
    # after preprocessing, ensure dense array for estimators that don't accept sparse
    pipe = Pipeline([
        ("pre", preprocessor),
        ("todense", FunctionTransformer(_to_dense)),
        ("clf", classifier),
    ])
    return pipe


def train_and_evaluate(X, y, test_size: float = 0.2, random_state: int = 42, classifier=None, **fit_kwargs) -> Tuple[Pipeline, Dict]:
    """Train the pipeline and return trained model and metrics dict.

    Metrics: accuracy and classification report (as string).
    This function attempts a stratified split; if stratification is not possible
    (too few samples per class), it falls back to a plain split and records a warning
    in the returned metrics.
    """
    stratify = None
    stratified_ok = False
    try:
        # try to stratify to preserve class balance in small datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        stratified_ok = True
    except ValueError:
        # fall back to non-stratified split (small dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    # determine numeric and categorical columns
    numeric_cols = list(X_train.select_dtypes(include=["number"]).columns)
    categorical_cols = list(X_train.select_dtypes(include=["object", "category"]).columns)

    # If no classifier provided, use default RandomForest
    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    pipe = build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols, classifier=classifier)
    pipe.fit(X_train, y_train, **fit_kwargs)

    preds = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    # If test set contains fewer than two classes, classification_report may be ill-defined.
    try:
        # avoid warnings when a class has no predicted samples
        report = classification_report(y_test, preds, zero_division=0)
    except Exception:
        report = "Classification report not available (test set contains fewer than 2 classes)."

    metrics = {
        "accuracy": acc,
        "report": report,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "stratified_split": bool(stratified_ok),
    }
    return pipe, metrics


def save_model(pipe: Pipeline, out_path: str):
    joblib.dump(pipe, out_path)


def load_model(path: str):
    return joblib.load(path)
