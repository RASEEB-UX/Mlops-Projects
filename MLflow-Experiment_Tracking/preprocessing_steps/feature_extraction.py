import pandas as pd

def temporal_features(df: pd.DataFrame, date_column: str):
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['quarter'] = df[date_column].dt.quarter
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['is_weekend'] = df[date_column].dt.dayofweek >= 5
    return df