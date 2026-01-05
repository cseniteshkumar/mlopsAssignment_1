"""Reusable stage functions for the ML pipeline.

Each function wraps existing project utilities so higher-level orchestration
can compose them without duplicating logic.
"""
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import json
import shutil

from src.data import load_dataset
from src.model import train_and_evaluate, save_model, load_model


def read_dataset(dataset_dir: str, target_col: Optional[str] = None):
    """Load dataset using project's data loader."""
    return load_dataset(dataset_dir, target_col=target_col)


def clean_data(X, y):
    """Lightweight cleaning using existing DataFrame ops (drop exact duplicates)."""
    df = X.copy()
    df["__target__"] = y.values
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if "__target__" in df.columns:
        y_clean = df["__target__"].copy()
        X_clean = df.drop(columns=["__target__"])
    else:
        X_clean = df
        y_clean = y
    return X_clean, y_clean, {"dropped_duplicates": int(before - after)}


def process_data(X):
    """Placeholder for dataset-level processing. Keep no-op to let pipeline's
    ColumnTransformer handle feature engineering.
    """
    return X


def validate_data(X, y) -> Dict[str, Any]:
    info = {}
    info["n_rows"] = int(len(X))
    info["n_cols"] = int(X.shape[1]) if hasattr(X, "shape") else None
    info["n_missing_cells"] = int(X.isnull().sum().sum()) if hasattr(X, "isnull") else None
    try:
        uniques = list(set(y.tolist()))
        info["n_classes"] = int(len(uniques))
        counts = {str(k): int((y == k).sum()) for k in uniques}
        info["class_counts"] = counts
    except Exception:
        info["n_classes"] = None
        info["class_counts"] = None
    issues = []
    if info["n_rows"] == 0:
        issues.append("no rows loaded")
    if info.get("n_classes") is not None and info["n_classes"] < 2:
        issues.append("target has fewer than 2 classes; cannot train classifier")
    return {"info": info, "issues": issues}


def train_model(X, y, test_size: float = 0.2, random_state: int = 42, classifier=None, **fit_kwargs):
    """Train the model using the existing `train_and_evaluate` helper.

    This function delegates splitting, training and evaluation to project utilities
    and returns the trained pipeline and metrics dict.
    """
    pipe, metrics = train_and_evaluate(X, y, test_size=test_size, random_state=random_state, classifier=classifier, **fit_kwargs)
    return pipe, metrics


def save_model_artifact(pipe, out_path: str, metadata: Optional[dict] = None):
    """Save the trained pipeline and optional metadata."""
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    save_model(pipe, str(outp))
    if metadata is not None:
        meta_path = outp.parent / "model_metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
    return str(outp)


def deploy_model(src_path: str, deploy_path: str = "models/model.joblib"):
    """Deploy model for API consumption by copying to `models/model.joblib`."""
    s = Path(src_path)
    d = Path(deploy_path)
    d.parent.mkdir(parents=True, exist_ok=True)
    # copy file byte-wise to preserve exact artifact
    shutil.copy2(str(s), str(d))
    return str(d)
