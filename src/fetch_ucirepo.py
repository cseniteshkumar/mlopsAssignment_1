"""Fetch a dataset from UCI ML Repository using ucimlrepo.fetch_ucirepo.

This script writes the primary CSV (if present) into the DataSet/ directory.
Usage: python src/fetch_ucirepo.py --id 45 --out DataSet
"""
import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, required=True, help="UCI repository dataset id (integer)")
    p.add_argument("--out", type=str, default="DataSet", help="Output folder to save CSV files")
    return p.parse_args(argv)


def run(args=None):
    if args is None:
        args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching dataset id={args.id} from UCI ML Repository...")
    # Special-case for the processed Cleveland heart-disease dataset (id=45)
    # Read directly from UCI using explicit column names and '?' as NA marker.
    if args.id == 45:
        import pandas as pd

        # URL for the processed Cleveland dataset
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        )

        # Define column names (as the raw file doesn't have a header)
        columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "num",
        ]

        # Read the data (dataset uses '?' for missing values)
        df = pd.read_csv(url, names=columns, na_values="?")

        out_path = out_dir / f"{args.id}_processed_cleveland.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved file: {out_path}")
        return

    # fetch_ucirepo accepts keyword 'id' or 'name' — pass id explicitly
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as e:
        raise RuntimeError("The package 'ucimlrepo' is required. Install it or update requirements.") from e

    data = fetch_ucirepo(id=args.id)

    # recursively search for pandas DataFrames inside the returned object
    import pandas as pd

    saved = []

    def _save_if_df(prefix, obj):
        # If obj is a DataFrame, save it. If dict-like, traverse.
        if isinstance(obj, pd.DataFrame):
            out_path = out_dir / f"{args.id}_{prefix}.csv"
            obj.to_csv(out_path, index=False)
            saved.append(str(out_path))
        elif hasattr(obj, "items"):
            for k, v in obj.items():
                _save_if_df(f"{prefix}_{k}", v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _save_if_df(f"{prefix}_{i}", v)

    # start from top-level
    if hasattr(data, "items"):
        for k, v in data.items():
            _save_if_df(k, v)
    else:
        _save_if_df(str(args.id), data)

    if not saved:
        meta_path = out_dir / f"{args.id}_metadata.json"
        meta_path.write_text(json.dumps({"data_repr": str(type(data))}))
        print("No DataFrame-like content found; saved metadata.")
    else:
        print(f"Saved files: {saved}")


if __name__ == "__main__":
    run()
