"""Reusable stage functions for the ML pipeline.

Each function wraps existing project utilities so higher-level orchestration
can compose them without duplicating logic.
"""
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import json
import shutil
import pandas as pd
import numpy as np

from src.data import load_dataset
from src.model import train_and_evaluate, save_model, load_model


def read_dataset(dataset_dir: str, target_col: Optional[str] = None):
    """Load dataset using project's data loader."""
    return load_dataset(dataset_dir, target_col=target_col)


def clean_data(X, y):
    """Comprehensive cleaning inspired by the provided notebook.

    - normalize column names (strip, lower, replace spaces)
    - replace '?' with NaN
    - attempt to coerce columns to numeric when majority of values parse
    - impute numeric columns with median and categorical with mode
    - drop exact duplicates and align target
    - binarize target if it appears to be the UCI 'num' multi-class target (>0 -> 1)
    """
    df = X.copy()
    # normalize column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # work on copy of target
    y_clean = y.copy()
    try:
        # binarize if name suggests UCI 'num' target or values >1 exist
        if getattr(y_clean, "name", None) and str(y_clean.name).strip().lower() == "num":
            y_clean = (y_clean > 0).astype(int)
        else:
            if pd.api.types.is_numeric_dtype(y_clean):
                if y_clean.max() is not None and float(y_clean.max()) > 1.0:
                    y_clean = (y_clean > 0).astype(int)
    except Exception:
        # keep original if anything goes wrong
        y_clean = y.copy()

    # replace common missing marker
    df = df.replace("?", np.nan)

    # attempt to coerce columns that are mostly numeric
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # if at least 60% of values parsed as numeric we keep the conversion
        non_null = coerced.notna().sum()
        if non_null >= max(1, int(0.6 * len(df))):
            df[col] = coerced

    # impute: numeric -> median, categorical -> mode
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                df[col] = df[col].fillna(median)
            else:
                if df[col].isnull().any():
                    mode = None
                    try:
                        mode = df[col].mode().iloc[0]
                    except Exception:
                        mode = "missing"
                    df[col] = df[col].fillna(mode)
        except Exception:
            # last-resort fill
            df[col] = df[col].fillna("missing")

    # drop duplicates and align target by index
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    dropped = int(before - after)

    # align y to df indices assuming original indexes were aligned
    try:
        y_aligned = y_clean.loc[df.index]
        # if lengths differ still, reset index alignment to positional
        if len(y_aligned) != len(df):
            y_aligned = y_clean.reset_index(drop=True).iloc[: len(df)]
    except Exception:
        y_aligned = y_clean.reset_index(drop=True).iloc[: len(df)]

    return df.reset_index(drop=True), y_aligned.reset_index(drop=True), {"dropped_duplicates": dropped}


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
