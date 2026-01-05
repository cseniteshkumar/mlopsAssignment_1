import os
from pathlib import Path

from src.pipeline import run_pipeline


def test_run_pipeline_creates_model(tmp_path):
    out = tmp_path / "model.joblib"
    result = run_pipeline(dataset="DataSet", target=None, output=str(out), test_size=0.1, random_state=0)
    assert out.exists(), f"Model file was not created at {out}"
    assert "test_metrics" in result
    # basic sanity on metrics structure when returned
    if result.get("test_metrics") is not None:
        assert isinstance(result.get("test_metrics"), dict)
