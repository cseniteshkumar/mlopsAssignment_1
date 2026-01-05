"""Fetch a dataset from UCI ML Repository using ucimlrepo.fetch_ucirepo.

This script writes the primary CSV (if present) into the DataSet/ directory.

"""


import pandas as pd

# URL for the processed Cleveland dataset
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)

# Define column names (as the raw file doesn't have a header)
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Read the data (dataset uses '?' for missing values)
df = pd.read_csv(url, names=columns, na_values="?")

out_path = "../DataSet/heart_disease_cleveland.csv"
df.to_csv(out_path, index=False)
print(f"Saved file: {out_path}")