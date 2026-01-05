"""Orchestrator for the end-to-end ML pipeline.

This module composes the stage functions defined in `src.stages` to avoid
duplicating logic that already exists in `src.data` and `src.model`.

It also optionally logs runs and artifacts to MLflow when available.
"""
from pathlib import Path
from typing import Optional, Dict, Any

import json
import pprint
try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    _MLFLOW_AVAILABLE = False

from src.stages import (
    read_dataset,
    clean_data,
    process_data,
    validate_data,
    train_model,
    save_model_artifact,
    deploy_model,
)


def run_pipeline(
    dataset: str = "DataSet",
    target: Optional[str] = None,
    output: str = "models/model.joblib",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    classifier=None,
    **clf_kwargs,
) -> Dict[str, Any]:
    """Run full pipeline: read -> clean -> process -> validate -> train -> store -> deploy.

    Uses shared stage functions so the logic is implemented in one place.
    """
    run_ctx = mlflow.start_run() if _MLFLOW_AVAILABLE else None
    try:
        # 1. Read
        X, y = read_dataset(dataset, target_col=target)

        # 2. Clean
        Xc, yc, clean_info = clean_data(X, y)

        # 3. Process (no-op; pipeline transformers handle feature work)
        Xp = process_data(Xc)

        # 4. Validate
        v = validate_data(Xp, yc)
        if v["issues"]:
            print("Validation issues found:", v["issues"])  # warn but continue

        # Log params to MLflow
        if _MLFLOW_AVAILABLE:
            try:
                mlflow.log_param("dataset", dataset)
                mlflow.log_param("target_col", str(target))
                mlflow.log_param("test_size", float(test_size))
                mlflow.log_param("val_size", float(val_size))
                mlflow.log_param("random_state", int(random_state))
            except Exception as e:
                print("Failed to log basic params to MLflow:", e)

        # 5/6. Train (delegate to shared helper which performs split/train/eval)
        pipe, train_metrics = train_model(Xp, yc, test_size=test_size, random_state=random_state, classifier=classifier, **clf_kwargs)

        # train_metrics contains evaluation on a test split
        test_metrics = train_metrics
        val_metrics = None
        if isinstance(train_metrics, dict) and "n_test" in train_metrics:
            # train_and_evaluate returns n_test; keep contract flexible
            pass

        # Log metrics to MLflow
        if _MLFLOW_AVAILABLE and isinstance(test_metrics, dict) and "accuracy" in test_metrics:
            try:
                mlflow.log_metric("test_accuracy", float(test_metrics["accuracy"]))
            except Exception:
                pass

        # 7. Store model and metadata
        meta = {"train_metrics": train_metrics, "clean_info": clean_info, "validate": v}
        saved_path = save_model_artifact(pipe, output, metadata=meta)

        # 8. Log model and artifacts to MLflow (if available)
        if _MLFLOW_AVAILABLE:
            try:
                mlflow.sklearn.log_model(pipe, artifact_path="model")
            except Exception as e:
                print("Failed to log model to MLflow:", e)
            try:
                outp = Path(saved_path)
                metrics_path = outp.parent / "pipeline_metrics.json"
                with open(metrics_path, "w") as fh:
                    json.dump(meta, fh, indent=2)
                mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
            except Exception as e:
                print("Failed to log artifacts to MLflow:", e)

        # 9. Deploy to API models/ location (backwards compatible)
        try:
            deployed = deploy_model(saved_path, deploy_path="models/model.joblib")
            print(f"Deployed model to {deployed}")
        except Exception as e:
            print("model_deploy failed:", e)

        return {"model_path": str(Path(saved_path)), "test_metrics": test_metrics, "val_metrics": val_metrics}
    finally:
        if _MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run end-to-end pipeline")
    p.add_argument("--dataset", default="DataSet")
    p.add_argument("--target", default=None)
    p.add_argument("--output", default="models/model.joblib")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()
    run_pipeline(dataset=args.dataset, target=args.target, output=args.output, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state)
