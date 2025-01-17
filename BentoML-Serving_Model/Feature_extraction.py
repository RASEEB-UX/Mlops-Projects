import pandas as pd
import numpy as np
import datetime as dt

def feature_extractor(data: pd.DataFrame) -> pd.DataFrame:
    
    """
    This function extracts time series features from the given data.
    """
    
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # Extract date components
    # data['hour'] = data['Date'].dt.hour
    data['dayofweek'] = data['Date'].dt.dayofweek
    data['quarter'] = data['Date'].dt.quarter
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    data['dayofyear'] = data['Date'].dt.dayofyear
    data['dayofmonth'] = data['Date'].dt.day
    data['weekofyear'] = data['Date'].dt.isocalendar().week
    
    
    # droping less imporatnt features
    data.drop(columns=['Date'], inplace=True)
    
    
    return data



