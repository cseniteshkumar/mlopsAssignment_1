"""Data loading utilities.

Loads all CSV files from a directory and returns feature matrix X and target y.
"""
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd


def load_dataset(dataset_dir: str, target_col: Optional[str] = "target") -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV files from dataset_dir and return X, y.

    Args:
        dataset_dir: path to a directory containing one or more CSV files.
        target_col: name of the target column in CSV files. If None, attempt to auto-detect.

    Returns:
        X: pandas DataFrame with features.
        y: pandas Series with target values.

    Raises:
        ValueError: if no CSV files are found or target column missing.
    """
    p = Path(dataset_dir)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    csv_files = sorted([f for f in p.iterdir() if f.suffix.lower() == ".csv"])
    if not csv_files:
        raise ValueError(f"No CSV files found in {dataset_dir}")
    # If target_col is None or set to 'auto', attempt to detect the target column.
    if target_col is None or str(target_col).lower() in ("auto", "detect"):
        # 1) Try to find a variables CSV that contains role==Target
        vars_candidates = [f for f in csv_files if "variable" in f.name.lower() or "variables" in f.name.lower()]
        detected = None
        for vf in vars_candidates:
            try:
                vdf = pd.read_csv(str(vf))
                if "role" in vdf.columns and "name" in vdf.columns:
                    trows = vdf[vdf["role"].str.lower() == "target"]
                    if not trows.empty:
                        detected = str(trows.iloc[0]["name"])
                        break
            except Exception:
                continue

        # 2) Fallback heuristics
        if detected is None:
            # common target column names
            common = ["target", "class", "label", "y", "num"]
            # look for any CSV that contains one of these columns
            for f in csv_files:
                try:
                    cols = pd.read_csv(str(f), nrows=0).columns.tolist()
                    for c in common:
                        if c in cols:
                            detected = c
                            break
                    if detected:
                        break
                except Exception:
                    continue

        if detected is None:
            raise ValueError(
                "Could not auto-detect target column. Provide target_col explicitly (e.g. --target num)."
            )
        target_col = detected

    # Only load CSV files that contain the target column to avoid mixing metadata files
    files_with_target = []
    for f in csv_files:
        try:
            cols = pd.read_csv(str(f), nrows=0).columns.tolist()
            if target_col in cols:
                files_with_target.append(f)
        except Exception:
            # ignore files that can't be read
            continue

    if not files_with_target:
        raise ValueError(f"Target column '{target_col}' not found in any CSV in {dataset_dir}")

    dfs = [pd.read_csv(str(f)) for f in files_with_target]
    df = pd.concat(dfs, ignore_index=True)

    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


if __name__ == "__main__":
    # quick local smoke test (won't run in tests)
    try:
        X, y = load_dataset("DataSet")
        print("Loaded", X.shape, y.shape)
    except Exception as e:
        print("Data load error:", e)
