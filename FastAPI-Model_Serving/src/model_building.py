import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names


rf = RandomForestClassifier()
rf.fit(X, y)

joblib.dump(rf, 'iris.pkl')
