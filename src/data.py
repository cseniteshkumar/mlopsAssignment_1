"""
Loads CSV file from a directory and returns feature matrix X and target y.
"""

import pandas as pd


def load_dataset():

    data = pd.read_csv("../DataSet/heart_disease_cleveland.csv")
    print(f"Data shape: {data.shape}")
    print(f"First 5 rows:\n{data.head()}")

    X = data.drop(columns=["num"])
    y = data["num"]
    return X, y

if __name__ == "__main__":
    try:
        X, y = load_dataset()
        print("\n\n\nLoaded", X.shape, y.shape)
    
    except Exception as e:
        print("\n\n\nData load error:", e)
