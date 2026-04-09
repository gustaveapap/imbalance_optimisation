
import numpy as np
import pandas as pd

def extract_features(df, target_quarter_hour):
    """
    Extract features for prediction from a DataFrame of system imbalance data.
    
    Args:
        df (pd.DataFrame): DataFrame with system imbalance data
        target_quarter_hour (pd.Timestamp): The quarter hour to predict
        
    Returns:
        np.array: Feature array of shape (60,) or None if insufficient data
    """
    feature_end = target_quarter_hour - pd.Timedelta(minutes=5)
    feature_start = feature_end - pd.Timedelta(minutes=60)
    
    feature_window = df[
        (df.index >= feature_start) & 
        (df.index < feature_end)
    ]['actual_system_imbalance'].values
    
    if len(feature_window) != 60:
        return None
        
    return feature_window

def prepare_features(df):
    """
    Prepare features for multiple quarter hours from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with system imbalance data
        
    Returns:
        tuple: (features array, target array, quarter hours array)
    """
    quarter_hours = sorted(df['quarter_hour'].unique())
    
    all_features = []
    all_targets = []
    all_times = []
    
    for qh in quarter_hours:
        features = extract_features(df, qh)
        if features is not None:
            all_features.append(features)
            target_value = df[df['quarter_hour'] == qh]['actual_system_imbalance'].mean()
            all_targets.append(target_value)
            all_times.append(qh)
    
    return (
        np.array(all_features),
        np.array(all_targets),
        np.array(all_times)
    )
