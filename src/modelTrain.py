"""
Model Training Module
This module handles the training of various machine learning models
on the heart disease dataset, utilizing MLflow for experiment tracking
and model management.
"""

from asyncio.log import logger
from pyexpat import model
import sys
import os

from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import mlflow.sklearn

experiment_name="Heart_Disease_Analysis"

mlflow_dir = "../mlruns"

mlflow_dir = Path(mlflow_dir)

if not os.path.exists(mlflow_dir):
    os.makedirs(mlflow_dir)

mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_dir)}")
mlflow.sklearn.autolog(log_models=True, log_datasets=True)
mlflow.set_experiment(experiment_name)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import recall_score, precision_score, f1_score

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from datetime import datetime


import joblib
import json
import psutil
import threading
import time
import tempfile
import logging

from sklearn.metrics import RocCurveDisplay



from src.dataRead import load_dataset
from src.dataClean import clean_data

output_dir = "../outputs/modelTrain"

models_dir = "../models"

models_dir = Path(models_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# Progress Bar
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable


# helper: monitor system metrics while a training call is running
def monitor_resources(stop_event, interval=0.5):
    samples = {"ts": [], "cpu": [], "mem": []}
    while not stop_event.is_set():
        samples["ts"].append(time.time())
        samples["cpu"].append(psutil.cpu_percent(interval=None))
        samples["mem"].append(psutil.virtual_memory().percent)
        stop_event.wait(interval)
    return samples
def start_monitoring():
    stop_event = threading.Event()
    results = {}
    def _runner():
        results.update(monitor_resources(stop_event))
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return stop_event, results, t
def stop_monitoring(stop_event, thread):
    stop_event.set()
    thread.join(timeout=5)
    # results dict will be filled (captured by closure)
    return


def modelTrain(data, model_path=None, generate_html=True):
    """
    Training the different model and storing for reference.     
    """
    df = data.copy()

    logger = logging.getLogger("modelTrain")
    logger.setLevel(logging.INFO)

    out_dir = Path(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Setup MLflow Experiment
    
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # persist scaler and feature list for inference pipeline
    try:
        joblib.dump(scaler, models_dir / "scaler.joblib")
        joblib.dump(X.columns.tolist(), models_dir / "features.joblib")
    except Exception:
        pass

    models = {
        "Logistic_Regression": LogisticRegression(),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
        "Naive_Bayes": GaussianNB()
    }

    # Start a Parent Run to group all model attempts together
    with mlflow.start_run(run_name="Algorithm_Comparison_Suite"):
        results = {}
        saved_artifacts = {}
        
        for name, model in models.items():
            safe_name = name.replace(" ", "_")
            # Start a Nested Child Run for each specific algorithm
            with mlflow.start_run(run_name=name, nested=True):
                # Feature: Autologging (Captures parameters and basic metrics automatically)
                mlflow.sklearn.autolog(log_models=True)

                # Tag the run with a clear model name for UI/search
                # mlflow.set_tag("model_name", name)
                

                mlflow.log_param("model_name", name)
                # start resource monitoring and timing
                stop_event, monitor_results, monitor_thread = start_monitoring()
                run_start = time.time()
                try:
                    model.fit(X_train_scaled, y_train.values)
                finally:
                    run_end = time.time()
                    stop_monitoring(stop_event, monitor_thread)
                duration = run_end - run_start
                # convert sampled series to lists (may be empty)
                cpu_series = monitor_results.get("cpu", []) if isinstance(monitor_results, dict) else []
                mem_series = monitor_results.get("mem", []) if isinstance(monitor_results, dict) else []
                ts_series = monitor_results.get("ts", []) if isinstance(monitor_results, dict) else []
                # aggregate system metrics
                if cpu_series:
                    cpu_avg = float(np.mean(cpu_series))
                    cpu_max = float(np.max(cpu_series))
                else:
                    cpu_avg = cpu_max = 0.0
                if mem_series:
                    mem_avg = float(np.mean(mem_series))
                    mem_max = float(np.max(mem_series))
                else:
                    mem_avg = mem_max = 0.0
                # Log duration and aggregated system metrics
                mlflow.log_metric("training_duration_sec", duration)
                mlflow.log_metric("sys_cpu_avg", cpu_avg)
                mlflow.log_metric("sys_cpu_max", cpu_max)
                mlflow.log_metric("sys_mem_avg", mem_avg)
                mlflow.log_metric("sys_mem_max", mem_max)
                # Save a trace (timestamps + samples) as an artifact
                trace = {
                    "model": name,
                    "start_ts": run_start,
                    "end_ts": run_end,
                    "duration_sec": duration,
                    "cpu_series": cpu_series,
                    "mem_series": mem_series,
                    "ts_series": ts_series
                }
                try:
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tf:
                        json.dump(trace, tf, indent=2)
                        tf_path = tf.name
                    mlflow.log_artifact(tf_path, artifact_path="traces")
                except Exception as e:
                    logger.warning("Failed to log trace artifact: %s", e)


                # predictions and evaluation
                predictions = model.predict(X_test_scaled)
                

                # predictions and evaluation
                predictions = model.predict(X_test_scaled)
                # try to get probabilistic scores for ROC/AUC
                probas = None
                try:
                    probas = model.predict_proba(X_test_scaled)[:, 1]
                except Exception:
                    try:
                        probas = model.decision_function(X_test_scaled)
                    except Exception:
                        probas = None


                # Feature: Custom Metrics (Crucial for medical data)
                acc = accuracy_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)


                # optionally log ROC AUC if scores available
                try:
                    if probas is not None:
                        auc = roc_auc_score(y_test, probas)
                        mlflow.log_metric("roc_auc", float(auc))
                    else:
                        auc = None
                except Exception:
                    auc = None
                
                # Manual logging for extra metrics not in autolog
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                
                # Feature: Tagging for easier searching
                mlflow.set_tag("model_family", "ensemble" if "Forest" in name else "linear")


                # Log confusion matrix and ROC plot as artifacts
                try:
                    cm = confusion_matrix(y_test, predictions)
                    fig_cm, ax_cm = plt.subplots(figsize=(4,4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    ax_cm.set_title(f"Confusion Matrix - {name}")
                    cm_path = out_dir / f"confusion_matrix_{safe_name}.png"
                    fig_cm.savefig(cm_path, bbox_inches="tight")
                    plt.close(fig_cm)
                    mlflow.log_artifact(str(cm_path), artifact_path="plots")
                except Exception as e:
                    logger.warning("Failed to save/log confusion matrix: %s", e)

                try:
                    if probas is not None:
                        fig_roc, ax_roc = plt.subplots(figsize=(6,4))
                        RocCurveDisplay.from_predictions(y_test, probas, ax=ax_roc)
                        ax_roc.set_title(f"ROC Curve - {name}")
                        roc_path = out_dir / f"roc_{safe_name}.png"
                        fig_roc.savefig(roc_path, bbox_inches="tight")
                        plt.close(fig_roc)
                        mlflow.log_artifact(str(roc_path), artifact_path="plots")
                except Exception as e:
                    logger.warning("Failed to save/log ROC plot: %s", e)



                # ensure the fitted model artifact is recorded (autolog usually does this,
                # but an explicit log helps if autolog missed it)
                # ensure the fitted model artifact is recorded (autolog usually does this,
                # but an explicit log helps if autolog missed it)
                try:
                    # register the model in the MLflow Model Registry using a safe name
                    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=safe_name)
                except Exception:
                    pass
                    # fallback: log without registering and add a tag so you can find it
                    try:
                        mlflow.sklearn.log_model(model, artifact_path=f"model_{safe_name}")
                        mlflow.set_tag("model_registered", "false")
                    except Exception:
                        # last-resort: continue without failing the whole loop
                        mlflow.set_tag("model_logging_failed", "true")
                
                results[name] = {"accuracy": acc, "recall": recall, "f1_score": f1}
                print(f"{name} | Acc: {acc:.2%} | Recall: {recall:.2%} | F1 Score: {f1:.2%}")

                # Save model artifact locally for future prediction pipeline
                try:
                    model_file = models_dir / f"{safe_name}.joblib"
                    joblib.dump(model, model_file)
                    saved_artifacts[name] = str(model_file)
                except Exception:
                    saved_artifacts[name] = None

        # Persist summary of results and saved artifact locations
        try:
            with open(models_dir / "metrics.json", "w") as f:
                json.dump({"results": results, "artifacts": saved_artifacts}, f, indent=2)
        except Exception:
            pass

    return results



if __name__ == "__main__":
    data = load_dataset()

    data = clean_data(data)
    result =  modelTrain(data)
    print("Training Result : "+str(result))
