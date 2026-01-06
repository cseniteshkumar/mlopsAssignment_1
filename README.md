# MLOps Assignment — Top-level overview

This repository is a compact, production-oriented scikit-learn pipeline for tabular classification (example: UCI Heart Disease). It demonstrates a complete ML lifecycle: data ingestion, cleaning, validation, EDA, training, model selection, deployment, batch & online prediction, and observability (MLflow). The design emphasizes reproducibility, simple artefact management, and clear separation between training and inference logic.

Contents (high level)
- src/
  - api.py — FastAPI inference server (health, file-predict endpoints)
  - dataRead.py — load CSVs from DataSet/ and return DataFrame(s)
  - dataClean.py — cleaning, imputation, simple outlier handling
  - dataValidation.py — domain checks / schema-like validations
  - dataEDA.py — EDA helpers and report export
  - fetch_ucirepo.py — helper to fetch example UCI datasets
  - modelTrain.py — core training, evaluation and MLflow logging
  - modelTrainPipeline.py — orchestrates full training pipeline
  - modelDeploy.py — selects & promotes best model for serving
  - modelBatchPrediction.py — batch prediction utility (CSV in → predictions out)
  - predict.py — simple CLI prediction for single files
- DataSet/ — input CSVs (user-supplied or fetched)
- models/ — output models & preprocessing artifacts
- outputs/ — reports, plots, metrics CSVs, exported transformed data
- mlruns/ (created by MLflow) — experiment runs and artifacts

Top-level workflow (end-to-end)
1. Data acquisition
   - Provide CSV files in DataSet/ (each file can be a partition). Optionally use src/fetch_ucirepo.py to fetch examples.
   - dataRead.py concatenates files and auto-detects common target names if not supplied.

2. Data cleaning & validation
   - Run dataClean.clean_data to perform median/mode imputations, type coercion, and light outlier handling.
   - Run dataValidation.validate_* checks to ensure values lie in expected ranges and required columns exist.

3. Exploratory Data Analysis (optional)
   - dataEDA produces quick plots, distributions, correlation matrices and an optional HTML report written to outputs/eda/.

4. Training & evaluation
   - modelTrain.py builds a preprocessing + classifier pipeline and evaluates with standard metrics (accuracy, precision, recall, f1, ROC AUC where applicable).
   - Supported classifiers: RandomForest (default), LogisticRegression, SVC.
   - Hyperparameter search: GridSearchCV, RandomizedSearchCV, and a manual randomized sampler with optional early stopping.
   - Cross-validation, model comparison, and export of transformed features are supported.

5. Model selection & deployment
   - modelDeploy.py contains simple logic to choose the best model (by metrics combination) and persist it as the deployed artifact (e.g., models/deployed_model.joblib).
   - Preprocessing artifacts (scaler, encoders, feature list) are saved and used at inference to ensure parity.

6. Prediction & serving
   - Batch: modelBatchPrediction.py reads CSV, applies preprocessing and writes predictions + probabilities (if available).
   - CLI: predict.py provides a small command-line wrapper for single-file predictions.
   - API: api.py exposes endpoints for health checks and file-based predictions; designed to load the deployed model artifact.

7. Observability & reproducibility
   - MLflow is integrated (default local file store: ./mlruns). The trainer logs parameters, metrics, artifacts (cv_results.csv, best_params.json, plots).
   - Environment flags:
     - ENABLE_MLFLOW=0 — disable MLflow logging
     - MLFLOW_EXPERIMENT — experiment name
     - MLFLOW_TRACKING_URI — custom MLflow tracking server

Quick start (minimal)
1. Create and activate virtualenv:
   - python3 -m venv .venv
   - source .venv/bin/activate
2. Install:
   - pip install -r requirements.txt
3. Add your CSV(s) into DataSet/ (or fetch example):
   - python src/fetch_ucirepo.py --id 45 --out DataSet
4. Train:
   - python src/modelTrainPipeline.py --dataset DataSet --target num --output models/model.joblib
   (or use modelTrain.py directly)
5. Deploy best model:
   - python src/modelDeploy.py --models-dir models --deployed models/deployed_model.joblib
6. Run batch predictions:
   - python src/modelBatchPrediction.py --model models/deployed_model.joblib --input DataSet/file.csv --output predictions.csv
7. Serve API:
   - uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Common CLI examples
- Train with grid search:
  - python src/modelTrain.py --dataset DataSet --target num --search grid --search-cv 5 --output models/best.joblib
- Randomized search with progress & patience:
  - python src/modelTrain.py --dataset DataSet --target num --search random --n-iter 50 --progress --patience 10 --output models/best_random.joblib
- Predict from CLI:
  - python src/predict.py --model models/final.joblib --input DataSet/new.csv --output predictions.csv

Artifacts produced
- models/*.joblib — serialized pipeline (preprocessing + model)
- models/features.joblib — list of features used at training time
- models/model_metadata.json — training metadata (params, version, timestamp)
- models/metrics.json or outputs/metrics_* — evaluation metrics
- outputs/eda/* — EDA plots and reports
- mlruns/* — MLflow runs and artifacts (if enabled)

Testing & CI
- Unit tests: python -m pytest -q
- A CI workflow (if present) runs tests and basic linting.
- Recommended: add tests for data validation, preprocessing parity, and an end-to-end inference smoke test.

Deployment notes & best practices
- Always save & load preprocessing artifacts (feature order, encoders, scalers).
- Use MLflow to track experiments and to store artifacts centrally when working in a team.
- For production serving:
  - Containerize with provided Dockerfile or Dockerfile.prod.
  - Use the deployed model artifact path known to the API (adjust _find_model_path in src/api.py if needed).
  - Add input schema validation (e.g., pandera or pydantic) to the API to fail fast on malformed inputs.

Troubleshooting
- Model not found in API: ensure models/deployed_model.joblib exists or update path in environment/API config.
- Prediction fails for missing columns: check models/features.joblib and ensure input CSV contains the same feature set (or implement fallback mapping).
- MLflow runs not visible: verify MLFLOW_TRACKING_URI or set ENABLE_MLFLOW=1.

Contributing & extensions
- Add stricter schema validation (pandera) and typed contracts for inputs/outputs.
- Add explainability artifacts (SHAP) and log them to MLflow.
- Implement model versioning and rollback in modelDeploy.py.
- Add automated end-to-end tests: training → deploy → inference.

License & contact
- See LICENSE in repository root for licensing details.
- See MODEL_CARD.md for intended use, limitations and data disclosure.

This README is intended as a single-pane topview. For implementation details, see docstrings and comments inside each file under src/.
```// filepath: /media/niteshkumar/SSD_Store_0_nvme/allPythoncodesWithPipEnv/BitsLearning/MLOps_Assignment/Assignment_1/README.md
// ...existing code...

# MLOps Assignment — Top-level overview

This repository is a compact, production-oriented scikit-learn pipeline for tabular classification (example: UCI Heart Disease). It demonstrates a complete ML lifecycle: data ingestion, cleaning, validation, EDA, training, model selection, deployment, batch & online prediction, and observability (MLflow). The design emphasizes reproducibility, simple artefact management, and clear separation between training and inference logic.

Contents (high level)
- src/
  - api.py — FastAPI inference server (health, file-predict endpoints)
  - dataRead.py — load CSVs from DataSet/ and return DataFrame(s)
  - dataClean.py — cleaning, imputation, simple outlier handling
  - dataValidation.py — domain checks / schema-like validations
  - dataEDA.py — EDA helpers and report export
  - fetch_ucirepo.py — helper to fetch example UCI datasets
  - modelTrain.py — core training, evaluation and MLflow logging
  - modelTrainPipeline.py — orchestrates full training pipeline
  - modelDeploy.py — selects & promotes best model for serving
  - modelBatchPrediction.py — batch prediction utility (CSV in → predictions out)
  - predict.py — simple CLI prediction for single files
- DataSet/ — input CSVs (user-supplied or fetched)
- models/ — output models & preprocessing artifacts
- outputs/ — reports, plots, metrics CSVs, exported transformed data
- mlruns/ (created by MLflow) — experiment runs and artifacts

Top-level workflow (end-to-end)
1. Data acquisition
   - Provide CSV files in DataSet/ (each file can be a partition). Optionally use src/fetch_ucirepo.py to fetch examples.
   - dataRead.py concatenates files and auto-detects common target names if not supplied.

2. Data cleaning & validation
   - Run dataClean.clean_data to perform median/mode imputations, type coercion, and light outlier handling.
   - Run dataValidation.validate_* checks to ensure values lie in expected ranges and required columns exist.

3. Exploratory Data Analysis (optional)
   - dataEDA produces quick plots, distributions, correlation matrices and an optional HTML report written to outputs/eda/.

4. Training & evaluation
   - modelTrain.py builds a preprocessing + classifier pipeline and evaluates with standard metrics (accuracy, precision, recall, f1, ROC AUC where applicable).
   - Supported classifiers: RandomForest (default), LogisticRegression, SVC.
   - Hyperparameter search: GridSearchCV, RandomizedSearchCV, and a manual randomized sampler with optional early stopping.
   - Cross-validation, model comparison, and export of transformed features are supported.

5. Model selection & deployment
   - modelDeploy.py contains simple logic to choose the best model (by metrics combination) and persist it as the deployed artifact (e.g., models/deployed_model.joblib).
   - Preprocessing artifacts (scaler, encoders, feature list) are saved and used at inference to ensure parity.

6. Prediction & serving
   - Batch: modelBatchPrediction.py reads CSV, applies preprocessing and writes predictions + probabilities (if available).
   - CLI: predict.py provides a small command-line wrapper for single-file predictions.
   - API: api.py exposes endpoints for health checks and file-based predictions; designed to load the deployed model artifact.

7. Observability & reproducibility
   - MLflow is integrated (default local file store: ./mlruns). The trainer logs parameters, metrics, artifacts (cv_results.csv, best_params.json, plots).
   - Environment flags:
     - ENABLE_MLFLOW=0 — disable MLflow logging
     - MLFLOW_EXPERIMENT — experiment name
     - MLFLOW_TRACKING_URI — custom MLflow tracking server

Quick start (minimal)
1. Create and activate virtualenv:
   - python3 -m venv .venv
   - source .venv/bin/activate
2. Install:
   - pip install -r requirements.txt
3. Add your CSV(s) into DataSet/ (or fetch example):
   - python src/fetch_ucirepo.py --id 45 --out DataSet
4. Train:
   - python src/modelTrainPipeline.py --dataset DataSet --target num --output models/model.joblib
   (or use modelTrain.py directly)
5. Deploy best model:
   - python src/modelDeploy.py --models-dir models --deployed models/deployed_model.joblib
6. Run batch predictions:
   - python src/modelBatchPrediction.py --model models/deployed_model.joblib --input DataSet/file.csv --output predictions.csv
7. Serve API:
   - uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Common CLI examples
- Train with grid search:
  - python src/modelTrain.py --dataset DataSet --target num --search grid --search-cv 5 --output models/best.joblib
- Randomized search with progress & patience:
  - python src/modelTrain.py --dataset DataSet --target num --search random --n-iter 50 --progress --patience 10 --output models/best_random.joblib
- Predict from CLI:
  - python src/predict.py --model models/final.joblib --input DataSet/new.csv --output predictions.csv

Artifacts produced
- models/*.joblib — serialized pipeline (preprocessing + model)
- models/features.joblib — list of features used at training time
- models/model_metadata.json — training metadata (params, version, timestamp)
- models/metrics.json or outputs/metrics_* — evaluation metrics
- outputs/eda/* — EDA plots and reports
- mlruns/* — MLflow runs and artifacts (if enabled)

Testing & CI
- Unit tests: python -m pytest -q
- A CI workflow (if present) runs tests and basic linting.
- Recommended: add tests for data validation, preprocessing parity, and an end-to-end inference smoke test.

Deployment notes & best practices
- Always save & load preprocessing artifacts (feature order, encoders, scalers).
- Use MLflow to track experiments and to store artifacts centrally when working in a team.
- For production serving:
  - Containerize with provided Dockerfile or Dockerfile.prod.
  - Use the deployed model artifact path known to the API (adjust _find_model_path in src/api.py if needed).
  - Add input schema validation (e.g., pandera or pydantic) to the API to fail fast on malformed inputs.

Troubleshooting
- Model not found in API: ensure models/deployed_model.joblib exists or update path in environment/API config.
- Prediction fails for missing columns: check models/features.joblib and ensure input CSV contains the same feature set (or implement fallback mapping).
- MLflow runs not visible: verify MLFLOW_TRACKING_URI or set ENABLE_MLFLOW=1.

Contributing & extensions
- Add stricter schema validation (pandera) and typed contracts for inputs/outputs.
- Add explainability artifacts (SHAP) and log them to MLflow.
- Implement model versioning and rollback in modelDeploy.py.
- Add automated end-to-end tests: training → deploy → inference.

License & contact
- See LICENSE in repository root for licensing details.
- See MODEL_CARD.md for intended use, limitations and data disclosure.

This README is intended as a single-pane topview. For implementation details, see docstrings and comments inside each file under src/.