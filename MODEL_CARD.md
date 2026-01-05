# Model card — scikit-learn RandomForest (default)

Model overview
- Model: scikit-learn pipeline (preprocessing + classifier)
- Default classifier: RandomForestClassifier
- Pipeline includes StandardScaler for numeric features and OneHotEncoder for categorical features.

Intended use
- Classification tasks on tabular data. Example: UCI Heart Disease dataset (id 45).

Metrics
- Training script saves metrics to `models/metrics.json` and logs to MLflow when enabled.

Reproducibility
- Model metadata saved to `models/model_metadata.json` including sklearn and numpy versions, model params, random_state and git commit (if available).

How to reproduce
1. Install dependencies: `pip install -r requirements.txt`
2. Place CSV(s) into `DataSet/` and run training, e.g.:
```bash
python src/train.py --dataset DataSet --target num --output models/model.joblib
```

Limitations
- The default pipeline performs one-hot encoding which can blow up dimensionality for very high-cardinality categorical features. Consider feature hashing or embedding for production.
- The trainer logs to MLflow by default (can be disabled). Local MLflow file store is used when no `MLFLOW_TRACKING_URI` is set.

Ethical considerations
- Ensure any dataset used respects privacy and consent. This code does not perform de-identification.
