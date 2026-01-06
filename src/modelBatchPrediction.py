"""
Batch prediction script
- Loads the deployed model from models/deployed_model.joblib (default)
- Loads a CSV dataset, drops the target column if present (configurable)
- Runs predictions (and probabilities if available)
- Saves predictions to outputs/ and logs artifact to MLflow
"""

from asyncio.log import logger
import json
import sys
import os

from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import mlflow
# import mlflow.sklearn

# experiment_name="Heart_Disease_Analysis"

# mlflow_dir = "../mlruns"

# mlflow_dir = Path(mlflow_dir)

# if not os.path.exists(mlflow_dir):
#     os.makedirs(mlflow_dir)

# mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_dir)}")
# mlflow.sklearn.autolog(log_models=True, log_datasets=True)
# mlflow.set_experiment(experiment_name)


# Progress Bar
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable


import argparse
import time
import pandas as pd
import joblib

batchProcessing = "../batchProcessing"

batchProcessing = Path(batchProcessing)

if not os.path.exists(batchProcessing):
    os.makedirs(batchProcessing)


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_and_save(model, X: pd.DataFrame, output_path: Path):
    preds = model.predict(X)
    out_df = pd.DataFrame({"prediction": preds}, index=X.index)

    # try to get probabilities if available
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            # if binary, take probability of positive class
            if proba.shape[1] == 2:
                out_df["probability"] = proba[:, 1]
            else:
                # add class-wise probabilities
                for i in range(proba.shape[1]):
                    out_df[f"prob_class_{i}"] = proba[:, i]
        except Exception:
            proba = None

    out_df.to_csv(output_path, index=False)
    return out_df


def batchPrediction(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    target_column: str = None
):
    model = load_model(model_path)
    data = pd.read_csv(data_path)

    if target_column and target_column in data.columns:
        X = data.drop(columns=[target_column])
    else:
        X = data

    predictions_df = predict_and_save(model, X, output_path)
    return predictions_df   

if __name__ == "__main__":

    model = Path("../models/deployed_model.joblib")
    data = batchProcessing / "batchData.csv"
    output = batchProcessing / "batchPrediction.csv"
    target_column = "num"

    batchPrediction(model, data, output, target_column)