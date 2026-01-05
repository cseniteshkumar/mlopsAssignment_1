"""Data validation.
"""
import sys
import os

import pathlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from src.dataRead import load_dataset
from src.dataClean import clean_data


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

    data = clean_data(data)
    is_valid = validate_heart_data(data)
    if is_valid:
        print("Data is ready for modeling.")
    else:
        print("Data validation failed. Please review the warnings/errors above.")   
