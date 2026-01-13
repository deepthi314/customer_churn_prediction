import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_tenure_groups(df):
    """
    Create categorical tenure groups from continuous tenure variable.
    Groups: 0-12, 12-24, 24-48, 48-60, 60+ months
    
    Args:
        df (pd.DataFrame): Dataframe with 'tenure' column.
        
    Returns:
        pd.DataFrame: Dataframe with new 'tenure_group' column.
    """
    logger.info("Creating tenure groups...")
    
    # Define bins and labels
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ['0-12', '12-24', '24-48', '48-60', '60+']
    
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    # One-hot encode the new group, or leave it for the main OHE pipeline?
    # Since this module is likely called BEFORE OHE in a real production pipeline or as part of it,
    # we'll just return the feature. The main preprocessing often assumes raw features.
    # However, if we want these specific groups to be OHE'd, we should ensure they are processed.
    # Let's keep it as a categorical column.
    
    return df

def calculate_clv(df):
    """
    Calculate Customer Lifetime Value (CLV).
    Proxy: tenure * MonthlyCharges
    
    Args:
        df (pd.DataFrame): Dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with 'CLV' column.
    """
    logger.info("Calculating CLV...")
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['CLV'] = df['tenure'] * df['MonthlyCharges']
    return df

def create_service_count(df):
    """
    Count the number of additional services a customer has.
    Services considered: PhoneService, MultipleLines, InternetService, 
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, 
    StreamingTV, StreamingMovies.
    
    Args:
        df (pd.DataFrame): Dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with 'ServiceCount' column.
    """
    logger.info("Creating ServiceCount feature...")
    services = [
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # We need to handle how these are encoded. 
    # If they are still 'Yes'/'No' or 1/0 (mapped in cleaning).
    # Cleaning mapped 'Yes'/'No' -> 1/0. 
    # 'No internet service' etc were left as object or need handling.
    # We should normalize them here temporarily to count.
    
    count_col = pd.Series(0, index=df.index)
    
    for col in services:
        if col in df.columns:
            # Check for 'Yes' string or 1 integer, or 'Yes' in object (if not yet mapped)
            # Safe way: convert to string and check if value == 'Yes' or value == '1'
             count_col += df[col].apply(lambda x: 1 if str(x).lower() in ['yes', '1'] else 0)
             
    df['ServiceCount'] = count_col
    return df

def create_interaction_features(df):
    """
    Create interaction features between key variables.
    - MonthlyCharges_per_Tenure = MonthlyCharges / (tenure + 1)
    
    Args:
        df (pd.DataFrame): Dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with interaction features.
    """
    logger.info("Creating interaction features...")
    
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['MonthlyCharges_per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        
    return df

def engineer_all_features(df):
    """
    Wrapper function to apply all feature engineering steps.
    
    Args:
        df (pd.DataFrame): Raw (cleaned) dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with all new features.
    """
    logging.info("Starting feature engineering pipeline...")
    df = df.copy()
    
    df = create_tenure_groups(df)
    df = calculate_clv(df)
    df = create_service_count(df)
    df = create_interaction_features(df)
    
    logging.info("Feature engineering pipeline completed.")
    return df

if __name__ == "__main__":
    # Example usage
    pass
