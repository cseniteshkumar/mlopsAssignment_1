import os
import sys
import tempfile
import subprocess

import pandas as pd
from sklearn.datasets import make_classification

from src.model import train_and_evaluate, save_model


def test_predict_script_runs_and_writes_csv():
    X, y = make_classification(n_samples=50, n_features=4, n_informative=2, n_redundant=0, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    # create temp dir
    with tempfile.TemporaryDirectory() as td:
        csv_in = os.path.join(td, "input.csv")
        df.to_csv(csv_in, index=False)

        # train a model on the same data (with target)
        df_train = df.copy()
        df_train["target"] = y
        model, metrics = train_and_evaluate(df_train.drop(columns=["target"]), df_train["target"], test_size=0.2)
        model_path = os.path.join(td, "model.joblib")
        save_model(model, model_path)

        # run predict.py using current Python interpreter
        out_path = os.path.join(td, "preds.csv")
        cmd = [sys.executable, "src/predict.py", "--model", model_path, "--input", csv_in, "--output", out_path]
        subprocess.check_call(cmd)

        assert os.path.exists(out_path)
        df_out = pd.read_csv(out_path)
        assert "prediction" in df_out.columns
        assert len(df_out) == len(df)
