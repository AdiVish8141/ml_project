import pandas as pd
import numpy as np

def load_petrol_data(filepath='data/ahmedabad_petrol_daily.csv'):
    """Loads, validates, and sorts historical Ahmedabad petrol price data."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['Date'])
    
    # Handle any missing values with forward-fill (no future lookahead)
    df['Petrol_Price'] = df['Petrol_Price'].ffill()
    df['Crude_Oil_USD'] = df['Crude_Oil_USD'].ffill()
    df['USD_INR'] = df['USD_INR'].ffill()
    df['Price_Change'] = df['Petrol_Price'].diff().fillna(0).round(2)
    
    return df
