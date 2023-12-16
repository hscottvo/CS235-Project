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
# max_depth_df = pd.DataFrame(columns=["implementation", "max_depth", "rmse"])
# for i in [5, 10, 15]:
#     scott_rf = RandomForest(n_estimators=10, max_depth=i)
#     sklearn_rf = RandomForestRegressor(n_estimators=10, max_depth=i)

#     scott_rf.fit(X_train.values, y_train.values)
#     sklearn_rf.fit(X_train.values, y_train.values)

#     scott_pred = scott_rf.predict(X_test.values)
#     sklearn_pred = sklearn_rf.predict(X_test.values)
#     rmse_scott = mean_squared_error(scott_pred, y_test, squared=False)
#     rmse_sklearn = mean_squared_error(sklearn_pred, y_test, squared=False)

#     new_data = pd.DataFrame(
#         {
#             "implementation": ["from_scratch", "sklearn"],
#             "max_depth": [i, i],
#             "rmse": [rmse_scott, rmse_sklearn],
#         }
#     )
#     max_depth_df = pd.concat([max_depth_df, new_data])
# print(max_depth_df)

scott_rf = RandomForest(n_estimators=3)
# sklearn_rf = RandomForestRegressor(n_estimators=10)

# scott_rf.fit(X_train.values, y_train.values)
# sklearn_rf.fit(X_train.values, y_train.values)

# scott_pred = scott_rf.predict(X_test.values)
# sklearn_pred = sklearn_rf.predict(X_test.values)
# rmse_scott = mean_squared_error(scott_pred, y_test, squared=False)
# rmse_sklearn = mean_squared_error(sklearn_pred, y_test, squared=False)

# print(rmse_scott)
# print(rmse_sklearn)
# print(scott_rf.depth())
