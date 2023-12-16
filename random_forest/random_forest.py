from statistics import mean

import numpy as np
import pandas as pd
from decision_tree import DecisionTree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_X_y
from tqdm import tqdm

import util


# main class for the estimator
class RandomForest:
    def __init__(
        self,
        n_estimators: int = 10,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int | None = None,
        max_depth: int | None = None,
        bootstrap: bool = True,
        max_samples: int | float | None = None,
    ):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_samples = max_samples

        if random_state:
            np.random.seed(random_state)

        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.estimators = []
        self.num_features = X.shape[1]
        X, y = check_X_y(X, y)
        if self.max_samples is None or self.bootstrap is False:
            max_samples = X.shape[0]
        elif isinstance(self.max_samples, int):
            max_samples = self.max_samples
            if max_samples > X.shape[0]:
                raise ValueError(
                    "max_samples can be at most the number of samples in X"
                )
        elif isinstance(self.max_samples, float):
            if self.max_samples > 1 or self.max_samples <= 0:
                raise ValueError(
                    "max_samples must be in (0.0, 1.0] if you pass in a float"
                )
            max_samples = max(round(X.shape[0] * self.max_samples), 1)
        else:
            raise TypeError("Must pass in None, ins, or float")
        for _ in tqdm(range(self.n_estimators)):
            curr_tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state,
                max_depth=self.max_depth,
            )
            indices = np.random.randint(X.shape[0], size=max_samples)
            X_sample = X[indices, :]
            y_sample = y[indices]
            curr_tree.fit(X_sample, y_sample)
            self.estimators.append(curr_tree)
        return self

    def predict(self, X: np.ndarray):
        if self.num_features != X.shape[1]:
            raise ValueError("Data does not match the training data shape")
        elif not self.estimators:
            raise RuntimeError("Random Forest has not been trained yet.")

        predictions = np.array(
            [i.evaluate(X) for i in self.estimators], dtype=np.float64
        )
        ret = predictions.mean(axis=0)
        return ret

    def score(self, X, y):
        # from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        pred = self.predict(X)
        rss = ((y - pred) ** 2).sum()
        tss = ((y - y.mean()) ** 2).sum()
        r_squared = 1 - (rss / tss)
        return r_squared

    def classify(self, X: np.ndarray):
        return np.sign(self.predict(X))

    def get_params(self, deep=False):
        return {"n_estimators": self.n_estimators}

    def depth(self):
        return max([i.depth for i in self.estimators])
