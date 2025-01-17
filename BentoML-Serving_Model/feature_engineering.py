import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# this function is used to remove outlier from Close price column
def remove_outliers(data:pd.DataFrame, col_name:str)->pd.DataFrame:
    q1 = data[col_name].quantile(0.25)
    q3 = data[col_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    data = data[(data[col_name] > lower_bound) & (data[col_name] < upper_bound)]
    return data

    
# this function perform log tranformation on Close column
def log_transform(data: pd.DataFrame) -> pd.DataFrame:
    data['Close'] = np.log(data['Close'])
    return data


# this function scales the selected features using StandardScaler
def scale_features(data: pd.DataFrame, features=None) -> pd.DataFrame:
    if features is None:
        features =[col for col in data.columns if col not in ['Date', 'Close']]
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data




