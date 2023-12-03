import json

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from decision_tree import DecisionTree, DecisionTreeNode, mse_split
from random_forest import RandomForest


# data = load_diabetes()
# data = fetch_california_housing()
class Data:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        pass


df = pd.DataFrame(
    {
        # "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "A": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "B": [2, 2, 2, 2, 2, 2, 2, 10, 10, 10],
        "C": [1, 2, 3, 4, 20, 22, 21, 90, 94, 99],
    }
)
data = Data(df[["A", "B"]].values, df[["C"]].values)
scott_tree = DecisionTree()
sklearn_tree = DecisionTreeRegressor()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=42, train_size=0.8
)

rf = RandomForest()
print(rf.x)

# print(X_train)
# scott_tree.fit(X_train, y_train)
# sklearn_tree.fit(X_train, y_train)
#
# scott_pred = scott_tree.evaluate(X_test)
#
# sklearn_pred = sklearn_tree.predict(X_test)
#
# diff = scott_pred - sklearn_pred
# print(diff)
# print(X_test)
# print(json.dumps(scott_tree.json(), indent=2))
