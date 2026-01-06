"""
Deploy the best (most recent) model from the experiment:
- Finds the latest run in the configured experiment
- Attempts to load the 'model' artifact via MLflow
- Runs predictions on the dataset (cleaned) and saves predictions
- Persists a deployed model file under models/ and logs artifacts to mlflow
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

# from src.modelTrain import monitor_resources, start_monitoring, stop_monitoring, modelTrain

from src.modelTrain import modelTrain

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


def modelDeploy():
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run(run_name="Model_Deployment", nested=True):

        # Persist deployed model to models_dir
        deployed_path = Path(models_dir) / "deployed_model.joblib"
        try:
            joblib.dump(loaded_model, deployed_path)
            mlflow.log_artifact(str(deployed_path))
            print(f"Deployed model saved to: {deployed_path}")
        except Exception as e:
            print("Failed to save deployed model:", e)        
        
    

if __name__ == "__main__":
    modelDeploy()