import pandas as pd
import numpy as np

def create_time_series_features(df):
    """Generates stationary lag features, rolling statistics, and delta targets.
    
    Predicting daily price return/change (Price_t - Price_{t-1}) ensures models
    (including Random Forest & XGBoost) generalize seamlessly to unseen price regimes
    without tree extrapolation failure.
    """
    df_feat = df.copy()
    
    # Calendar features
    df_feat['DayOfWeek'] = df_feat['Date'].dt.dayofweek
    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['Quarter'] = df_feat['Date'].dt.quarter
    
    # Lag Features for Petrol Price (Levels)
    df_feat['Petrol_Lag_1'] = df_feat['Petrol_Price'].shift(1)
    df_feat['Petrol_Lag_2'] = df_feat['Petrol_Price'].shift(2)
    df_feat['Petrol_Lag_7'] = df_feat['Petrol_Price'].shift(7)
    df_feat['Petrol_Lag_14'] = df_feat['Petrol_Price'].shift(14)
    df_feat['Petrol_Lag_30'] = df_feat['Petrol_Price'].shift(30)
    
    # Differences / Returns
    df_feat['Petrol_Diff_1'] = df_feat['Petrol_Lag_1'] - df_feat['Petrol_Lag_2']
    df_feat['Petrol_Diff_7'] = df_feat['Petrol_Lag_1'] - df_feat['Petrol_Lag_7']
    
    # Exogenous Macro Lags
    df_feat['Crude_Lag_1'] = df_feat['Crude_Oil_USD'].shift(1)
    df_feat['Crude_Diff_1'] = df_feat['Crude_Oil_USD'].shift(1) - df_feat['Crude_Oil_USD'].shift(2)
    df_feat['Crude_Diff_7'] = df_feat['Crude_Oil_USD'].shift(1) - df_feat['Crude_Oil_USD'].shift(7)
    
    df_feat['USD_Lag_1'] = df_feat['USD_INR'].shift(1)
    df_feat['USD_Diff_1'] = df_feat['USD_INR'].shift(1) - df_feat['USD_INR'].shift(2)
    
    # Rolling Statistics
    df_feat['Petrol_Rolling_Mean_7'] = df_feat['Petrol_Lag_1'].rolling(window=7).mean()
    df_feat['Petrol_Rolling_Std_7'] = df_feat['Petrol_Lag_1'].rolling(window=7).std().fillna(0)
    
    # Target: Actual current price AND Delta target
    df_feat['Target_Price'] = df_feat['Petrol_Price']
    df_feat['Target_Delta'] = df_feat['Petrol_Price'] - df_feat['Petrol_Lag_1']
    
    # Drop initial NaN rows
    df_feat = df_feat.dropna().reset_index(drop=True)
    
    feature_cols = [
        'DayOfWeek', 'Month', 'Quarter',
        'Petrol_Diff_1', 'Petrol_Diff_7',
        'Crude_Lag_1', 'Crude_Diff_1', 'Crude_Diff_7',
        'USD_Lag_1', 'USD_Diff_1',
        'Petrol_Rolling_Std_7'
    ]
    
    return df_feat, feature_cols

