"""Exploratory data analysis utilities.

Generates histograms for numeric features, a correlation heatmap, and class balance plot.
Saves PNGs to the output directory.

"""

import sys
import os

import pathlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from src.dataRead import load_dataset
from src.dataClean import clean_data

output_dir = "../outputs/modelTrain"

def modelTrain(data, model_path=None, generate_html=True):
    """
    
    """
    df = data.copy()

    out_dir = Path(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    
# import mlflow
# import mlflow.sklearn
# from sklearn.metrics import recall_score, precision_score, f1_score

# def train_and_evaluate_with_mlflow(df, experiment_name="Heart_Disease_Analysis"):
#     # 1. Setup MLflow Experiment
#     mlflow.set_experiment(experiment_name)
    
#     X = df.drop('target', axis=1)
#     y = df['target']
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     models = {
#         "Logistic_Regression": LogisticRegression(),
#         "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#         "SVM": SVC(probability=True),
#         "KNN": KNeighborsClassifier(n_neighbors=5)
#     }

#     # Start a Parent Run to group all model attempts together
#     with mlflow.start_run(run_name="Algorithm_Comparison_Suite"):
#         results = {}
        
#         for name, model in models.items():
#             # Start a Nested Child Run for each specific algorithm
#             with mlflow.start_run(run_name=name, nested=True):
#                 # Feature: Autologging (Captures parameters and basic metrics automatically)
#                 mlflow.sklearn.autolog(log_models=True)
                
#                 model.fit(X_train_scaled, y_train)
#                 predictions = model.predict(X_test_scaled)
                
#                 # Feature: Custom Metrics (Crucial for medical data)
#                 acc = accuracy_score(y_test, predictions)
#                 recall = recall_score(y_test, predictions)
#                 f1 = f1_score(y_test, predictions)
                
#                 # Manual logging for extra metrics not in autolog
#                 mlflow.log_metric("accuracy", acc)
#                 mlflow.log_metric("recall", recall)
#                 mlflow.log_metric("f1_score", f1)
                
#                 # Feature: Tagging for easier searching
#                 mlflow.set_tag("model_family", "ensemble" if "Forest" in name else "linear")
                
#                 results[name] = acc
#                 print(f"{name:<20} | Acc: {acc:.2%} | Recall: {recall:.2%}")

#     return results

# # Execution
# results = train_and_evaluate_with_mlflow(cleaned_df)




if __name__ == "__main__":
    data = load_dataset()

    data = clean_data(data)
    modelTrain(data)
