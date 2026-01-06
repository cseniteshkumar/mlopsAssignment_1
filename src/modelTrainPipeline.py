"""
Model Training pipeline
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



from src.dataRead import load_dataset
from src.dataClean import clean_data
from src.modelTrain import modelTrain
from src.modelDeploy import modelDeploy


models_dir = "../models"

models_dir = Path(models_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Progress Bar
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable


def trainingPipeline():
    with mlflow.start_run(run_name="Training Pipeline", nested=True ):

        data = load_dataset()

        data = clean_data(data)
        traingResult = modelTrain(data)
        deployedModel = modelDeploy()

        mlflow.log_param("Data set", data)
        mlflow.log_metric("random_state : ", int(42))
        mlflow.log_param("Dataset Shape", str(data.shape))
        mlflow.log_dict(traingResult, "traingResult.json")
        mlflow.log_param("Model Deployed : ", str(deployedModel))
        
        print(traingResult)

if __name__ == "__main__":
    trainingPipeline()