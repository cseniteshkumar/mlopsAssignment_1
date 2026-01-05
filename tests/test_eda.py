import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd


def test_eda_generates_images(tmp_path):
    # create synthetic dataset
    df = pd.DataFrame({
        "a": range(50),
        "b": [x % 5 for x in range(50)],
        "target": [0 if x < 25 else 1 for x in range(50)],
    })
    ds_dir = tmp_path / "DataSet"
    ds_dir.mkdir()
    csv_path = ds_dir / "data.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "eda_out"
    cmd = [sys.executable, "src/eda.py", "--dataset", str(ds_dir), "--target", "target", "--out", str(out_dir)]
    res = subprocess.run(cmd, cwd=Path.cwd())
    assert res.returncode == 0

    # check that at least one histogram and class balance exist
    assert (out_dir / "hist_a.png").exists()
    assert (out_dir / "class_balance.png").exists()
