"""Data validation.
"""

import pandas as pd
import numpy as np

from src.data import load_dataset



def clean_data(data):
    """
    Handles loading, missing values, and type conversion.
    """

    df = data.copy()   

    # Ensure consistent data types
    df['ca'] = df['ca'].astype(int)
    df['thal'] = df['thal'].astype(int)

    # Handle Missing Values (Imputation)
    # Instead of dropping, we fill 'ca' and 'thal' with their median 
    # This prevents losing valuable data from the other 11 columns
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].median())

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
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Binary Target Conversion
    df['target'] = (df['target'] > 0).astype(int)
 
    return df


def validate_heart_data(df):
    """
    Performs logical checks to ensure data integrity.
    Returns True if valid, raises an error or prints warnings otherwise.
    """
    validation_passed = True
    
    # Check 1: No missing values
    if df.isnull().values.any():
        print("❌ Validation Failed: Missing values detected.")
        validation_passed = False
        
    # Check 2: Logical ranges for medical vitals
    # (e.g., Blood pressure 'trestbps' usually between 80 and 200)
    if not df['trestbps'].between(50, 250).all():
        print("⚠️ Validation Warning: Outlier detected in Blood Pressure.")
        
    # Check 3: Categorical consistency
    # sex should be 0 or 1; cp (chest pain) should be 1, 2, 3, or 4
    if not df['sex'].isin([0, 1]).all():
        print("❌ Validation Failed: Unexpected values in 'sex' column.")
        validation_passed = False
        
    # Check 4: Data volume
    if len(df) < 200:
        print("⚠️ Validation Warning: Dataset size is smaller than expected.")

    if validation_passed:
        print("✅ Data Validation Successful: Dataset is clean and logical.")
    
    return validation_passed



if __name__ == "__main__":
    data = load_dataset()

    print("Original data shape:", data.shape)

    data = clean_data(data)
    print("Cleaned data shape:", data.shape)
    is_valid = validate_heart_data(data)
    if is_valid:
        print("Data is ready for modeling.")
    else:
        print("Data validation failed. Please review the warnings/errors above.")   
        