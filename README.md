# MLOps Assignment — scikit-learn pipeline

This repository contains a reproducible scikit-learn training pipeline that:

- loads tabular CSV data from `DataSet/` (the helper loader will auto-detect the target column when possible),
- performs preprocessing (missing-value imputation, scaling for numeric features, one-hot encoding for categoricals),
- trains and evaluates models (RandomForest, LogisticRegression, SVC supported),
- supports hyperparameter search (GridSearchCV, RandomizedSearchCV, and manual randomized sampling with progress/early-stopping),
- logs experiments and artifacts to MLflow (local file store by default),
- provides EDA tooling and exports artifacts (plots, HTML report),
- includes tests and small utilities (UCI fetch helper).

## Quick start

1. Create and activate a virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put CSV files into the `DataSet/` directory. Each CSV should include the same set of feature columns; at least one file must contain the target column. If you don't know the target name, omit `--target` and the loader will try to auto-detect (looks for a `variables` CSV and common names like `target`, `class`, `label`, `num`).

4. Fetch UCI dataset (optional):

```bash
python src/fetch_ucirepo.py --id 45 --out DataSet
```

5. Run training (example):

```bash
python src/train.py --dataset DataSet --target num --output models/model.joblib
```

## Useful commands and examples

- Run EDA and generate plots:

```bash
python src/eda.py --dataset DataSet --target num --out outputs/eda --html
```

- Train with a hyperparameter grid search (GridSearchCV):

```bash
python src/train.py --dataset DataSet --target num --output models/best.joblib --search grid --search-cv 5 --classifier rf
```

- Randomized search with progress and early stopping (manual sampler):

```bash
python src/train.py --dataset DataSet --target num --output models/best_random.joblib --search random --n-iter 50 --progress --patience 10
```

- Compare models (LogisticRegression vs RandomForest) via cross-validation and export transformed features:

```bash
python src/train.py --dataset DataSet --target num --output models/final.joblib --compare-models --export-features
```

- Make predictions with a saved model:

```bash
python src/predict.py --model models/final.joblib --input DataSet/some_file.csv --output predictions.csv
```

## MLflow

- By default MLflow uses a local file store at `./mlruns`. To inspect runs:

```bash
mlflow ui
# then open http://127.0.0.1:5000
```

- Environment variables:
  - `ENABLE_MLFLOW=0` — disable MLflow logging
  - `MLFLOW_EXPERIMENT` — experiment name (default: `default`)
  - `MLFLOW_TRACKING_URI` — if set, trainer will use this tracking URI

- The training CLI uploads `cv_results.csv` and `best_params.json` when a hyperparameter search is used, and can create nested MLflow child runs for each candidate (`--mlflow-child-runs`).

## Files and tools

- `src/data.py` — dataset loader that concatenates CSVs in `DataSet/` and returns `X, y`.
- `src/model.py` — builds preprocessing + classifier pipeline and contains train/evaluate helpers.
- `src/train.py` — main CLI for training, search, MLflow logging, model comparison and feature export.
- `src/eda.py` — EDA utilities (histograms, correlation heatmap, pairplots, per-class distributions, optional feature importances and simple HTML report).
- `src/fetch_ucirepo.py` — helper to download datasets from the UCI ML repository via `ucimlrepo`.
- `src/predict.py` — load a saved model and write predictions to CSV.
- `tests/` — pytest-based tests. Run them with `python -m pytest`.

## Model selection & evaluation

- The preprocessing is: numeric median imputation + StandardScaler, categorical most-frequent imputation + OneHotEncoder.
- Models supported: RandomForest (default), LogisticRegression, SVC.
- Evaluation metrics logged and reported: accuracy, precision (macro), recall (macro), ROC AUC (binary uses `roc_auc`, multiclass uses `roc_auc_ovr`).
- Cross-validation is used for model comparison (`--compare-models`) and for hyperparameter search (controlled with `--search` and `--search-cv`).
- The trainer saves artifacts into the same directory as the `--output` model (e.g., `models/`): `metrics.json`, `model_metadata.json`, `cv_results.csv`, `best_params.json`, `model_comparison.json`, `model_selection.md`, `transformed_features.csv` (when exported).

## Tests

- Run the test suite with:

```bash
python -m pytest -q
```

## Notes & next steps

- The repo includes a `Dockerfile` and a GitHub Actions workflow to run the tests. If you want, I can also package the project into a simple CLI distribution or add nested CV for model selection.

If you'd like the README to include screenshots of EDA artifacts or a short tutorial notebook, tell me and I will generate them.
# MLOps Assignment — scikit-learn pipeline

This project contains a small scikit-learn based training pipeline that reads CSV files from the `DataSet/` folder, trains a classifier, evaluates it, and saves a trained model to `models/`.

Quick start

1. Install dependencies (preferably inside a virtualenv):

```bash
pip install -r requirements.txt
```

2. Put CSV files into the `DataSet/` directory. Each CSV should include a target column. If you don't know the target column name, you can omit `--target` and the loader will attempt to auto-detect it (it looks for a `variables` CSV and common names like `target`, `class`, `label`, `num`).

3. (Optional) You can fetch UCI repository datasets directly using `ucimlrepo` and the included helper script. Example (dataset id 45 — Heart Disease):

```bash
python src/fetch_ucirepo.py --id 45 --out DataSet
```

4. Run training (example):

```bash
python src/train.py --dataset DataSet --target num --output models/model.joblib
```

Or let the loader auto-detect the target:

```bash
python src/train.py --dataset DataSet --output models/model.joblib
```

What was implemented

- `src/data.py`: utilities to load CSV files from `DataSet/` and return X, y
- `src/model.py`: builds a preprocessing + classifier pipeline and trains/evaluates
- `src/train.py`: CLI wrapper to run training and save model
- `tests/`: unit tests for the data loader and a small train integration test

Assumptions

- CSVs in `DataSet/` contain a column named `target` by default; you can override with `--target`.
- This is implemented using scikit-learn and uses a RandomForestClassifier by default.

MLflow & advanced options

- To view MLflow runs (local file-store), start the MLflow UI in the repository root:

```bash
mlflow ui
```

Then open http://127.0.0.1:5000 to see runs, parameters, metrics, and artifacts. (By default MLflow stores artifacts and run data in `./mlruns`.)

- To disable MLflow logging set `ENABLE_MLFLOW=0` in the environment. By default logging is enabled.
- To set a custom experiment name: `export MLFLOW_EXPERIMENT=my_experiment`.
- To set a remote MLflow server use `export MLFLOW_TRACKING_URI=<uri>`.

Classifier selection

- Use `--classifier` to choose between `rf` (RandomForest), `logreg` (LogisticRegression) or `svm` (SVC). Example:

```bash
python src/train.py --dataset DataSet --classifier logreg --C 0.5 --output models/model_lr.joblib
```

The trainer will log ROC AUC and ROC plots to MLflow when available.

```
# MLOps Assignment — scikit-learn pipeline

This project contains a small scikit-learn based training pipeline that reads CSV files from the `DataSet/` folder, trains a classifier, evaluates it, and saves a trained model to `models/`.

Quick start

1. Install dependencies (preferably inside a virtualenv):

```bash
pip install -r requirements.txt
```

2. Put CSV files into the `DataSet/` directory. Each CSV should include a target column. If you don't know the target column name, you can omit `--target` and the loader will attempt to auto-detect it (it looks for a `variables` CSV and common names like `target`, `class`, `label`, `num`).

3. (Optional) You can fetch UCI repository datasets directly using `ucimlrepo` and the included helper script. Example (dataset id 45 — Heart Disease):

```bash
python src/fetch_ucirepo.py --id 45 --out DataSet
```

4. Run training (example):

```bash
python src/train.py --dataset DataSet --target num --output models/model.joblib
```

Or let the loader auto-detect the target:

```bash
python src/train.py --dataset DataSet --output models/model.joblib
```

What was implemented

- `src/data.py`: utilities to load CSV files from `DataSet/` and return X, y
- `src/model.py`: builds a preprocessing + classifier pipeline and trains/evaluates
- `src/train.py`: CLI wrapper to run training and save model
- `tests/`: unit tests for the data loader and a small train integration test

Assumptions

- CSVs in `DataSet/` contain a column named `target` by default; you can override with `--target`.
- This is implemented using scikit-learn and uses a RandomForestClassifier by default.
