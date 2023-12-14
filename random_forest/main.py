import json

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from decision_tree import DecisionTree, DecisionTreeNode, mse_split
from random_forest import RandomForest


class Data:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        pass


df = pd.read_csv("../train.csv")
data = Data(df.drop(columns="profit_margin"), df["profit_margin"])

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=42, train_size=0.8
)

args = {"n_estimators": 10, "max_depth": 5}

scott_rf = RandomForest(**args)
sklearn_rf = RandomForestRegressor(**args)

scott_rf.fit(X_train.values, y_train.values)
sklearn_rf.fit(X_train.values, y_train.values)

scott_pred = scott_rf.predict(X_test.values)
sklearn_pred = sklearn_rf.predict(X_test.values)

print(mean_squared_error(scott_pred, sklearn_pred, squared=True))
