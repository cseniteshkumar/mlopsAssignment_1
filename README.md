

# MLOps Assignment — Quick reference

This repository is a compact scikit-learn pipeline for tabular classification (example: UCI Heart Disease). It includes data loading, cleaning, training, model selection, deployment and simple serving utilities with optional MLflow tracking.

Important source files
- Training pipeline: [`modelTrainPipeline.trainingPipeline`](src/modelTrainPipeline.py) 
- Core trainer: [`modelTrain.modelTrain`](src/modelTrain.py) 
- Model selection & promotion: [`modelDeploy.modelDeploy`](src/modelDeploy.py)
- Batch predictions: [`modelBatchPrediction.batchPrediction`](src/modelBatchPrediction.py)
- API (serve): [src/api.py](src/api.py) — UVicorn entrypoint: `src.api:app`


Run locally (development)
1. Create virtualenv and install deps:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train (simple):
```sh
python src/modelTrainPipeline.py 
```
This invokes [`modelTrainPipeline.trainingPipeline`](src/modelTrainPipeline.py) which calls [`modelTrain.modelTrain`](src/modelTrain.py).

3. Deploy best model:
```sh
python src/modelDeploy.py 
```
This runs [`modelDeploy.modelDeploy`](src/modelDeploy.py) and writes the deployed artifact to `models/`.

4. Batch prediction:
```sh
python src/modelBatchPrediction.py 

Uses [`modelBatchPrediction.batchPrediction`](src/modelBatchPrediction.py).

5. Serve API (local):
```sh
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Send file-based inference requests to the API endpoints defined in [src/api.py](src/api.py).

Production deployment (container / Kubernetes)
- Build production image (uses provided Dockerfile.prod):
```sh
docker build -f Dockerfile.prod -t my-ml-service:latest .
```
- Run container:
```sh
docker build -t mlops-assignment:prod -f Dockerfile.prod .
docker run --rm -p 8080:8080 --name mlops-prod mlops-assignment:prod
```

Where to look for implementation details
- Data loading & cleaning: [src/dataRead.py](src/dataRead.py), [src/dataClean.py](src/dataClean.py)
- Training & evaluation internals: [src/modelTrain.py](src/modelTrain.py)
- Model promotion & simple selection logic: [src/modelDeploy.py](src/modelDeploy.py)
- Batch inference implementation: [src/modelBatchPrediction.py](src/modelBatchPrediction.py)
- API endpoints & serving code: [src/api.py](src/api.py)
