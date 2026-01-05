"""Simple prediction script.

Usage:
    python src/predict.py --model models/model.joblib --input some.csv --output preds.csv

If --output is omitted prints predictions to stdout.
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import sys
from pathlib import Path

# Ensure project root on sys.path so joblib can unpickle pipeline steps defined in `src`.
_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to saved model (joblib)")
    p.add_argument("--input", required=True, help="CSV file with features (no target column required)")
    p.add_argument("--output", required=False, help="CSV path to write predictions")
    return p.parse_args(argv)


def run(args=None):
    if args is None:
        args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    model = joblib.load(str(model_path))
    X = pd.read_csv(args.input)

    # If input contains a target column accidentally, drop it
    for c in ["target", "label", "class", "y", "num"]:
        if c in X.columns:
            X = X.drop(columns=[c])

    preds = model.predict(X)
    df_out = pd.DataFrame({"prediction": preds})

    if args.output:
        df_out.to_csv(args.output, index=False)
        print(f"Wrote predictions to {args.output}")
    else:
        print(df_out.to_csv(index=False))


if __name__ == "__main__":
    run()
