import bentoml
import xgboost as xgb
import numpy as np
import pandas as pd
from feature_engineering import log_transform, scale_features
from Feature_extraction import feature_extractor
from data_ingest import get_data

@bentoml.service
class StockPredictionService:
    def __init__(self):
        self.model = bentoml.xgboost.load_model("stock_prediction_model:latest")

    def preprocess(self, data: pd.DataFrame) -> np.ndarray:
        # Apply feature extraction
        features = feature_extractor(data)
        scaled_features = scale_features(features)
        return scaled_features

    def prepare_input(self, date_str: str):
        # Convert the date string into a DataFrame
        data = pd.DataFrame({'Date': [date_str]})
        # Fetch additional data if necessary
        # Example: data = get_data('SPY', '2023-01-01', '2023-01-02')
        return data

    @bentoml.api
    def predict(self, input_data) -> np.ndarray:
        if isinstance(input_data, str):
            # If input is a date string, convert it to DataFrame
            data = self.prepare_input(input_data)
        elif isinstance(input_data, pd.DataFrame):
            data = input_data
        else:
            raise ValueError("Input must be a date string or a pandas DataFrame.")

        preprocessed_data = self.preprocess(data)
        dmatrix = xgb.DMatrix(preprocessed_data)
        predictions = self.model.predict(dmatrix)
        return np.expm1(predictions)

# Example usage
if __name__ == "__main__":
    # Example with a date string
    service = StockPredictionService()
    predictions = service.predict("2019-01-01")
    print(predictions)


# Important Links

# https://medium.com/@rizqi.okta/bentoml-is-all-you-need-creating-machine-learning-service-into-deployment-with-ease-367f440f0a05

# https://docs.bentoml.com/en/latest/examples/xgboost.html