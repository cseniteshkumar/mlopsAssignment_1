import numpy as np
import pandas as pd
from scipy import sparse

from src.model import _to_dense, build_pipeline, train_and_evaluate


def test_to_dense_with_sparse_matrix():
    mat = sparse.csr_matrix([[1, 0], [0, 2]])
    out = _to_dense(mat)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)


def test_build_pipeline_numeric_only():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    pipe = build_pipeline(numeric_cols=["a", "b"], categorical_cols=[], classifier=None)
    # Should be able to fit on numeric data
    y = np.array([0, 1, 0])
    pipe.fit(df, y)
    preds = pipe.predict(df)
    assert len(preds) == 3


def test_train_and_evaluate_stratify_fallback():
    # create a dataset with only one class in y (stratify would fail)
    df = pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 0, 0, 0]})
    X = df[["x"]]
    y = df["target"]
    model, metrics = train_and_evaluate(X, y, test_size=0.5)
    assert "accuracy" in metrics
    assert metrics["n_train"] + metrics["n_test"] == 4
