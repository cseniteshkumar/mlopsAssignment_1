import os
import tempfile

import pandas as pd
import pytest

from src.data import load_dataset


def test_load_dataset_missing_dir_raises():
    with pytest.raises(ValueError):
        load_dataset("nonexistent_dir_abc123")


def test_load_dataset_no_csvs_raises(tmp_path):
    empty = tmp_path / "emptydir"
    empty.mkdir()
    with pytest.raises(ValueError):
        load_dataset(str(empty))


def test_load_dataset_auto_detect_target(tmp_path):
    # create variables file indicating target
    vars_df = pd.DataFrame({"name": ["f1", "target"], "role": ["feature", "target"]})
    vars_dir = tmp_path / "DataSet"
    vars_dir.mkdir()
    (vars_dir / "variables.csv").write_text(vars_df.to_csv(index=False))

    # create data CSV containing detected target
    data_df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})
    (vars_dir / "data.csv").write_text(data_df.to_csv(index=False))

    X, y = load_dataset(str(vars_dir), target_col=None)
    assert "target" not in X.columns
    assert len(y) == 2
