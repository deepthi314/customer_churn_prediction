import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from src.data_preprocessing import clean_data, load_data

def test_pipeline():
    try:
        df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        print(f"Original Churn values: {df['Churn'].unique()}")
        
        df_clean = clean_data(df)
        print(f"Cleaned Churn values: {df_clean['Churn'].unique()}")
        
        # Check if Churn is all NaNs
        if df_clean['Churn'].isnull().all():
            print("ISSUE CONFIRMED: Churn is all NaNs due to double mapping.")
        else:
            print("Churn looks okay.")

        # Check shapes
        y = df_clean['Churn']
        print(f"y shape: {y.shape}")
        print(f"y ndim: {y.ndim}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_pipeline()
