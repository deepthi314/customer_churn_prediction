import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Load data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_data(df):
    """
    Perform initial data cleaning.
    - Drop customerID
    - Convert TotalCharges to numeric
    - Handle missing values
    - Encode binary variables
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    logger.info("Starting data cleaning...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Drop customerID as it's not predictive
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    # TotalCharges is object, convert to numeric. Coerce errors to NaN (empty strings become NaN)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
    # Handle missing values in TotalCharges (usually for tenure=0)
    missing_tc = df['TotalCharges'].isnull().sum()
    if missing_tc > 0:
        logger.info(f"Found {missing_tc} missing values in TotalCharges. Filling with 0.")
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
    # Encode binary categorical variables "Yes"/"No"
    # Identify columns with "Yes"/"No" values
    for col in df.columns:
        if col == 'Churn':
            continue
            
        if df[col].dtype == 'object':
            unique_vals = set(df[col].dropna().unique())
            if unique_vals == {'Yes', 'No'}:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
            elif unique_vals == {'Yes', 'No', 'No internet service'} or unique_vals == {'Yes', 'No', 'No phone service'}:
                 # For now, we keep these as object for OHE or specialized encoding later, 
                 # or map 'No internet service' to 'No' depending on strategy.
                 # Strategy: Replace "No internet service" with "No" for simpler binary encoding? 
                 # The prompt suggests "Create binary features". 
                 # For simplicity and robust OHE later, we might leave them or map them.
                 # Let's map "No ... service" to "No" for cleaner binary features if desired, 
                 # but full OHE is better for tree models. 
                 # Let's LEAVE them for OHE in the pipeline.
                 pass
                 
    # Target variable 'Churn'
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    logger.info("Data cleaning completed.")
    return df

def preprocess_pipeline(filepath, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline.
    
    Args:
        filepath (str): Path to CSV.
        test_size (float): Proportion of test set.
        random_state (int): Random seed.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler, features
    """
    df = load_data(filepath)
    df = clean_data(df)
    
    target = 'Churn'
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in dataset.")
    
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # One-Hot Encoding
    logger.info("Performing One-Hot Encoding...")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Train-test split
    logger.info(f"Splitting data with test_size={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale numerical features
    logger.info("Scaling numerical features...")
    scaler = StandardScaler()
    
    # We need to scale only the original numerical columns, NOT the dummy variables typically,
    # OR scale everything. Scaling everything is safer for some models (SVM, LR).
    # Since we have dummy variables (0/1), scaling them is debatable but often fine/recommended for regularization.
    # However, for interpretability, let's strictly scale the continuous vars.
    # But X_train now has different columns.
    
    # Re-identifying numerical columns after OHE is tricky if we want to only scale original numerics.
    # Simpler approach: Scale ONLY known continuous columns: 'tenure', 'MonthlyCharges', 'TotalCharges'
    continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # Check if they exist in X_train (they might have been dropped or renamed by accident, though unlikely)
    cols_to_scale = [col for col in continuous_cols if col in X_train.columns]
    
    if cols_to_scale:
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    features = X_train.columns.tolist()
    
    logger.info("Preprocessing pipeline completed.")
    return X_train, X_test, y_train, y_test, scaler, features

if __name__ == "__main__":
    # Example usage (commented out requires data)
    # try:
    #     X_train, X_test, y_train, y_test, scaler, features = preprocess_pipeline('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    #     print("Preprocessing successful!")
    #     print(f"Train shape: {X_train.shape}")
    #     print(f"Test shape: {X_test.shape}")
    # except Exception as e:
    #     print(f"Error: {e}")
    pass
