"""Airflow-friendly wrappers that call the project's stage functions and
persist intermediate artifacts to disk so tasks can be chained without
passing large in-memory objects through XCom.

These functions are intentionally small and idempotent: they create an
`artifacts/` directory by default and write files with predictable names.
"""
from pathlib import Path
import json
import joblib

from src.stages import (
    read_dataset,
    clean_data,
    process_data,
    validate_data,
    train_model,
    save_model_artifact,
    deploy_model,
)


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def read_stage(dataset_dir: str = "DataSet", target_col: str = None, out_dir: str = "artifacts") -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = read_dataset(dataset_dir, target_col=target_col)
    x_path = out / "raw_X.pkl"
    y_path = out / "raw_y.pkl"
    joblib.dump(X, x_path)
    joblib.dump(y, y_path)
    return str(out)


def clean_stage(artifacts_dir: str = "artifacts") -> str:
    out = Path(artifacts_dir)
    X = joblib.load(out / "raw_X.pkl")
    y = joblib.load(out / "raw_y.pkl")
    Xc, yc, info = clean_data(X, y)
    joblib.dump(Xc, out / "clean_X.pkl")
    joblib.dump(yc, out / "clean_y.pkl")
    with open(out / "clean_info.json", "w") as fh:
        json.dump(info, fh, default=str)
    return str(out)


def process_stage(artifacts_dir: str = "artifacts") -> str:
    out = Path(artifacts_dir)
    Xc = joblib.load(out / "clean_X.pkl")
    Xp = process_data(Xc)
    joblib.dump(Xp, out / "processed_X.pkl")
    return str(out)


def validate_stage(artifacts_dir: str = "artifacts") -> str:
    out = Path(artifacts_dir)
    Xp = joblib.load(out / "processed_X.pkl")
    yc = joblib.load(out / "clean_y.pkl")
    v = validate_data(Xp, yc)
    with open(out / "validate.json", "w") as fh:
        json.dump(v, fh, default=str)
    return str(out)


def train_stage(artifacts_dir: str = "artifacts", output_model: str = "models/model.joblib", classifier=None, **clf_kwargs) -> str:
    out = Path(artifacts_dir)
    Xp = joblib.load(out / "processed_X.pkl")
    yc = joblib.load(out / "clean_y.pkl")
    pipe, metrics = train_model(Xp, yc, classifier=classifier, **clf_kwargs)
    # Ensure model dir
    model_path = Path(output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    with open(out / "train_metrics.json", "w") as fh:
        json.dump(metrics, fh, default=str)
    return str(model_path)


def deploy_stage(model_path: str, deploy_path: str = "models/model.joblib") -> str:
    # Reuse deploy_model helper which may perform copying or other ops
    deployed = deploy_model(model_path, deploy_path=deploy_path)
    return str(deployed)
