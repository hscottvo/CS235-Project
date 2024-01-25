import numpy as np
import progressbar

from decision_tree import RegressionTree
from utils import squared_error, huber, bar_widgets

class GradientBoostingRegressor(object):
    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, loss):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.loss = loss
        
        if loss == 'squared_error':
            self.loss = squared_error()
        elif loss == 'huber':
            self.loss = huber()

        self.bar = progressbar.ProgressBar(widgets=bar_widgets) # progress bar

        # create n_estimators trees
        self.trees = []
        for _ in range(n_estimators):
            tree = RegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_impurity,
                    max_depth=self.max_depth,
                    loss=self.loss)
            self.trees.append(tree)


    def fit(self, X, y):
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for i in self.bar(range(self.n_estimators)):
            if i == 0:
                self.trees[i].fit(X, y)
                y_pred = self.trees[i].predict(X)
            else:
                gradient = self.loss.loss(y, y_pred)
                self.trees[i].fit(X, gradient)
                update = self.trees[i].predict(X)
                y_pred += np.multiply(self.learning_rate, update)   # update y prediction by multiplying with learning rate

    def predict(self, X):
        y_pred = np.array([])
        for index, tree in enumerate(self.trees):
            if index == 0:  # if on first tree, y_pred is just the predicted values from tree
                update = tree.predict(X)
                y_pred = update
            else:           # otherwise add learning rate * predicted values
                update = tree.predict(X)
                update = np.multiply(self.learning_rate, update)
                y_pred += update 

        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        sum_squared_regression = ((y - y_pred) ** 2).sum()
        total_sum_squares = ((y - y.mean()) ** 2).sum()
        r_squared = 1 - (sum_squared_regression / total_sum_squares)
        return r_squared

    def get_params(self, deep=False):
        return {'n_estimators':self.n_estimators, 'learning_rate':self.learning_rate, 
                'min_samples_split':self.min_samples_split, 'min_impurity':self.min_impurity, 
                'max_depth':self.max_depth, 'loss':self.loss}
