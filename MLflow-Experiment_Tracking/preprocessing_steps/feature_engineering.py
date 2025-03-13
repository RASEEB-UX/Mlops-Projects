import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import numpy as np

def bool_encoding(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)  # Convert boolean to integer (True -> 1, False -> 0)
    return df

def hot_encode(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:
        if df[col].dtype.name == 'category' and df[col].nunique() < 10:
            df = pd.get_dummies(df, columns=[col], prefix=[col])
    return df



def binary_encode(df: pd.DataFrame) -> pd.DataFrame:
    
    for col in df.columns:
        if df[col].dtype.name == 'category' and df[col].nunique() >= 10:
            encoder = ce.BinaryEncoder(cols=[col])
            df = encoder.fit_transform(df)
    return df


def feature_scaling(df: pd.DataFrame):
  
    for col in df.columns:
        if df[col].dtype in ['int64', 'int32', 'float64'] and col != 'exchange_rate':
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
    return df

def log_transform(df: pd.DataFrame, target_col: str):
    df[target_col] = df[target_col].apply(lambda x: np.log1p(x))
    return df


def dtype_conversion(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'date':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'object' and col == 'date':
            df[col] = pd.to_datetime(df[col])
    return df