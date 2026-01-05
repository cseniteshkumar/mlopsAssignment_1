import os
import tempfile

import pandas as pd

from src.data import load_dataset


def test_load_dataset_basic():
    # create temp directory and CSV
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sample.csv")
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4],
            "f2": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        })
        df.to_csv(path, index=False)

        X, y = load_dataset(td, target_col="target")
        assert "target" not in X.columns
        assert len(X) == 4
        assert len(y) == 4
