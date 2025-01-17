import typing as t

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch

import bentoml

if t.TYPE_CHECKING:
    from sklearn.utils import Bunch

    from bentoml._internal import external_typing as ext

# Load the data
cancer: Bunch = t.cast("Bunch", load_breast_cancer())
cancer_data = t.cast("ext.NpNDArray", cancer.data)
cancer_target = t.cast("ext.NpNDArray", cancer.target)
dt = xgb.DMatrix(cancer_data, label=cancer_target)

# Specify model parameters
param = {"max_depth": 3, "eta": 0.3, "objective": "multi:softprob", "num_class": 2}

# Train the model
model = xgb.train(param, dt)

# Specify the model name and the model to be saved
bentoml.xgboost.save_model("cancer", model)

import os

import numpy as np
import xgboost as xgb

import bentoml


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class CancerClassifier:
    # Retrieve the latest version of the model from the BentoML model store
    bento_model = bentoml.models.get("cancer:latest")

    def __init__(self):
        self.model = bentoml.xgboost.load_model(self.bento_model)

        # Check resource availability
        if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
            self.model.set_param({"predictor": "gpu_predictor", "gpu_id": 0})  # type: ignore (incomplete XGBoost types)
        else:
            nthreads = os.getenv("OMP_NUM_THREADS")
            if nthreads:
                nthreads = max(int(nthreads), 1)
            else:
                nthreads = 1
            self.model.set_param({"predictor": "cpu_predictor", "nthread": nthreads})

    @bentoml.api
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(xgb.DMatrix(data))
    

import xgboost as xgb

import bentoml

# Load the model by setting the model tag
booster = bentoml.xgboost.load_model("cancer:latest")

# Predict using a sample
res = booster.predict(
    xgb.DMatrix(
        [
            [
                1.308e01,
                1.571e01,
                8.563e01,
                5.200e02,
                1.075e-01,
                1.270e-01,
                4.568e-02,
                3.110e-02,
                1.967e-01,
                6.811e-02,
                1.852e-01,
                7.477e-01,
                1.383e00,
                1.467e01,
                4.097e-03,
                1.898e-02,
                1.698e-02,
                6.490e-03,
                1.678e-02,
                2.425e-03,
                1.450e01,
                2.049e01,
                9.609e01,
                6.305e02,
                1.312e-01,
                2.776e-01,
                1.890e-01,
                7.283e-02,
                3.184e-01,
                8.183e-02,
            ]
        ]
    )
)

print(res)
    
