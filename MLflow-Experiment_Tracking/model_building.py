from sklearn.model_selection import train_test_split
import pandas as pd
from data_ingest import read_csv
from preprocessing_steps.feature_extraction import temporal_features
from preprocessing_steps.feature_engineering import dtype_conversion, bool_encoding, binary_encode, feature_scaling, log_transform
from preprocessing_steps.remove_columns import del_columns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings
warnings.filterwarnings("ignore")


mlflow.set_tracking_uri("sqlite:///mlflow_1.db")
mlflow.set_experiment("Exchange Rate Prediction")

df = read_csv('data_source/daily_forex_rates.csv')

df = dtype_conversion(df)

# keeping the data within 80th percentile of exchange rate
nth_percentile = 0.80
df_filtered = df[df['exchange_rate'] <= df['exchange_rate'].quantile(nth_percentile)]


columns_to_delete = ['currency_name', 'date', 'base_currency']
df_filtered = temporal_features(df_filtered, 'date')
df_filtered = binary_encode(df_filtered)
df_filtered = bool_encoding(df_filtered)
df_filtered = feature_scaling(df_filtered)
df_filtered = log_transform(df_filtered, 'exchange_rate')
df_filtered = del_columns(df_filtered, columns_to_delete)

df_filtered.head()
# time to split the dataset into train and test set
X = df_filtered.drop('exchange_rate', axis=1)
y = df_filtered['exchange_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# ------------------------- Experiment) 1: Random Forest Regressor -----------


with mlflow.start_run():
    rfr = RandomForestRegressor(n_estimators=100, max_depth=10,
                                random_state=42)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.sklearn.log_model(rfr, "random_forest_model")

# ------------------------- Experiment) 2: xgboost Regressor -----------------


with mlflow.start_run():
    xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.01,
                               reg_alpha=0.05, reg_lambda=0.03,
                               random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("reg_alpha", 0.05)
    mlflow.log_param("reg_lambda", 0.03)
    mlflow.log_param("model", "xgboost")
    mlflow.xgboost.log_model(xgb, "xgboost_regressor")

# Run this whole scripts before executing below command in terminal to see the results in UI

# mlflow ui --backend-store-uri sqlite:///mlflow_1.db