"""
Model Training Module
This module handles the training of various machine learning models
on the heart disease dataset, utilizing MLflow for experiment tracking
and model management.
"""

from pyexpat import model
import sys
import os

import pathlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import mlflow.sklearn

# Call this BEFORE importing sklearn estimators or metrics
mlflow.sklearn.autolog(log_models=True)

# mlflow.sklearn.autolog(log_datasets=False)
mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_dir)}")
mlflow.sklearn.autolog(log_models=True, log_datasets=True)
mlflow.set_experiment(experiment_name)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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




experiment_name="Heart_Disease_Analysis"

from src.dataRead import load_dataset
from src.dataClean import clean_data

output_dir = "../outputs/modelTrain"

mlflow_dir = "../mlruns"

mlflow_dir = Path(mlflow_dir)

if not os.path.exists(mlflow_dir):
    os.makedirs(mlflow_dir)


models_dir = "../models"

models_dir = Path(models_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)



# Progress Bar
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable


def modelTrain(data, model_path=None, generate_html=True):
    """
    Training the different model and storing for reference.     
    """
    df = data.copy()

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
                
                # model.fit(X_train_scaled, y_train)
                model.fit(X_train_scaled, y_train.values)
                predictions = model.predict(X_test_scaled)
                
                # Feature: Custom Metrics (Crucial for medical data)
                acc = accuracy_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)
                
                # Manual logging for extra metrics not in autolog
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                
                # Feature: Tagging for easier searching
                mlflow.set_tag("model_family", "ensemble" if "Forest" in name else "linear")


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
