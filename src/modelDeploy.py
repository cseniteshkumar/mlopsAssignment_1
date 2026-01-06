"""
Deploy the best (most recent) model from the experiment:
- Finds the latest run in the configured experiment
- Attempts to load the 'model' artifact via MLflow
- Runs predictions on the dataset (cleaned) and saves predictions
- Persists a deployed model file under models/ and logs artifacts to mlflow
"""

from asyncio.log import logger
import json
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

import shutil


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


def select_best_balanced_model(results_data):
    results = results_data['results']
    
    # Define a custom ranking rule: Average of Accuracy and F1 Score
    def ranking_score(model_name):
        metrics = results[model_name]
        return (metrics['accuracy'] + metrics['f1_score']) / 2

    # Find the model name with the highest average score
    best_model_name = max(results.keys(), key=ranking_score)
    
    return {
        'model_name': best_model_name,
        'combined_score': ranking_score(best_model_name),
        'metrics': results[best_model_name],
        'path': results_data['artifacts'][best_model_name]
    }


def modelDeploy():
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run(run_name="Model_Deployment", nested=True):
        # refer to the experiment
        results_data_path = Path(models_dir) / "metrics.json"
        results_data = json.load(open(results_data_path, 'r'))
        
        # mlflow.log_metric("Load Metrics", str(results_data))
        mlflow.log_dict(results_data, "model_results.json")

        best_model_info = select_best_balanced_model(results_data)
        model_name = best_model_info['model_name']
        model_path = best_model_info['path']
        combined_score = best_model_info['combined_score']
        metrics = best_model_info['metrics']

        mlflow.log_param("Best Model Name ", str(model_name))
        mlflow.log_param("Best Model Path ", str(model_path))
        mlflow.log_param("Best Model Combined Score ", str(combined_score))
        mlflow.log_param("Best Model Metrics ", str(metrics))


        print(f"Deployed Model: {model_name} from {model_path}")
        
        sourcePath = Path(model_path)
        deployed_path = Path(models_dir) / "deployed_model.joblib"
        # deployed_path.parent.mkdir(parents=True, exist_ok=True)
        # copy file byte-wise to preserve exact artifact
        shutil.copy2(str(sourcePath), str(deployed_path))
        print(f"Deployed model saved to: {deployed_path}")

   

if __name__ == "__main__":
    modelDeploy()