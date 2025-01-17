import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from data_ingest import get_data
from feature_engineering import log_transform, scale_features
from Feature_extraction import feature_extractor
import datetime as dt
import optuna
import bentoml

def apply_all_steps(symbol, start_date, end_date):
    data = get_data(symbol, start_date, end_date)
    data = feature_extractor(data)
    data = log_transform(data)
    data = scale_features(data)
    return data

def objective(trial):
    df = apply_all_steps('SPY', '2015-01-01', dt.date.today())
    X = df.drop(['Close','Open','High','Low','Volume'], axis=1)
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_loguniform("eta", 0.01, 0.3),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0)
    }

    model = xgb.train(param, dtrain, num_boost_round=100)
    predictions = model.predict(dtest)
    mse = mean_squared_error(y_test, predictions)
    return mse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

best_params = study.best_params
df = apply_all_steps('SPY', '2015-01-01', dt.date.today())
X = df.drop(['Close','Open','High','Low','Volume'], axis=1)
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

model = xgb.train(best_params, dtrain, num_boost_round=100)

# Evaluate model
predictions = model.predict(dtest)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")



bentoml.xgboost.save_model("stock_prediction_model", model)



