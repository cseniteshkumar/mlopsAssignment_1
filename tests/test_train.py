import os
import tempfile

import pandas as pd
from sklearn.datasets import make_classification

from src.model import train_and_evaluate, save_model


def test_train_and_save_model():
    # ensure n_informative + n_redundant <= n_features
    X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, n_classes=2, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "data.csv")
        df.to_csv(csv_path, index=False)

        # train
        model, metrics = train_and_evaluate(df.drop(columns=["target"]), df["target"], test_size=0.2)
        assert "accuracy" in metrics

        # save
        out_path = os.path.join(td, "model.joblib")
        save_model(model, out_path)
        assert os.path.exists(out_path)
