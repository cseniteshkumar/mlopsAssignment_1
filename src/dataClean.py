"""Data validation.
"""
import sys
import os

import pathlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from src.dataRead import load_dataset



def clean_data(data):
    """
    Handles loading, missing values, and type conversion.
    """

    df = data.copy()   

    # Handle Missing Values (Imputation)
    # Instead of dropping, we fill 'ca' and 'thal' with their median 
    # This prevents losing valuable data from the other 11 columns
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].median())

   # Ensure consistent data types
    df['ca'] = df['ca'].astype(int)
    df['thal'] = df['thal'].astype(int)

    # # Remove rows with missing values
    df = df.dropna()


    # Handle Outliers (Capping)
    # Cholesterol values > 500 or BP > 200 are rare and can skew models.
    # We clip them to reasonable medical upper bounds.
    df['chol'] = df['chol'].clip(upper=500)
    df['trestbps'] = df['trestbps'].clip(upper=200)

    
    # Type Standardization
    int_cols = ['age', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    df[int_cols] = df[int_cols].astype(int)

    
    # Standardize Target (0 = Healthy, 1 = Disease)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    
    # Binary Target Conversion
    df['target'] = (df['target'] > 0).astype(int)
 
    #### Delete original target column
    df = df.drop(columns=['num'], axis=1)

    return df


if __name__ == "__main__":
    data = load_dataset()

    print("Original data shape:", data.shape)

    data = clean_data(data)
    print("Cleaned data shape:", data.shape)
 
