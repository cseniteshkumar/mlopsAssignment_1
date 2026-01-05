import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd


def test_train_with_grid_search(tmp_path):
    # small synthetic dataset
    df = pd.DataFrame({
        "feat1": [0, 1, 0, 1, 0, 1, 0, 1],
        "feat2": [1, 2, 1, 2, 1, 2, 1, 2],
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    ds_dir = tmp_path / "DataSet"
    ds_dir.mkdir()
    csv_path = ds_dir / "data.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "models"
    out_dir.mkdir()
    model_path = out_dir / "model_search.joblib"

    cmd = [
        sys.executable,
        "src/train.py",
        "--dataset",
        str(ds_dir),
        "--target",
        "target",
        "--output",
        str(model_path),
        "--search",
        "grid",
        "--search-cv",
        "2",
        "--classifier",
        "rf",
    ]

    res = subprocess.run(cmd, cwd=Path.cwd())
    assert res.returncode == 0

    # check outputs
    assert model_path.exists()
    assert (out_dir / "best_params.json").exists()
    assert (out_dir / "metrics.json").exists()
