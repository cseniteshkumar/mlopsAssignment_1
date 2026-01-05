import pandas as pd
from src.model import train_and_evaluate


def test_preprocessing_handles_numeric_and_categorical():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0],
        "cat1": ["a", "b", "a", "c"],
        "target": [0, 1, 0, 1],
    })

    X = df.drop(columns=["target"])
    y = df["target"]

    model, metrics = train_and_evaluate(X, y, test_size=0.5)
    # Ensure model trained and returned metrics
    assert "accuracy" in metrics
    assert metrics["n_train"] + metrics["n_test"] == 4
