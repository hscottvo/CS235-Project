import numpy as np

from utils import mse, divide_on_feature

class DecisionNode():
    def __init__(self, feature_i=None, threshold=None,
                 value=None, left_branch=None, right_branch=None, data=None):
        self.feature_i = feature_i          # index for the feature that is tested
        self.threshold = threshold          # threshold value for feature
        self.value = value                  # predicted value (residual)
        self.left_branch = left_branch      # left subtree
        self.right_branch = right_branch    # right subtree
        self.data = data                    # data array inside the leaf node


class RegressionTree(object):
    def __init__(self, min_samples_split = 2, min_impurity = 1e-7, max_depth = 3, loss = None):
        self.root = None
        self.min_samples_split = min_samples_split          # min samples required to split
        self.min_impurity = min_impurity                    # min impurity to justify split
        self.max_depth = max_depth                          # max tree depth 
        self.impurity_calculation = self.impurity_decrease  # function to calculate impurity
        self.leaf_value_calculation = self.mean_of_y        # function to determine prediction of y at leaf
        self.loss = loss                           

    # build tree
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        self.impurity_calculation = self.impurity_decrease
        self.leaf_value_calculation = self.mean_of_y

    def impurity_decrease(self, y, y1, y2):
        curr_impurity = mse(y)
        l_impurity = mse(y1)
        r_impurity = mse(y2)
        l_num = len(y1) / len(y)
        r_num = len(y2) / len(y)
        impurity = curr_impurity - (l_impurity * l_num + r_impurity * r_num) # formula from sklearn
        return impurity

    def mean_of_y(self, y): 
        # calculate the mean of the values in the leaf node 
        avg = np.mean(y)
        return np.mean(avg)

    def build_tree(self, X, y, current_depth=0):
        largest_impurity = 0
        best_criteria = None    # feature index and threshold
        best_sets = None        # subsets of the data

        # check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # add y as last column of X
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # calculate the impurity for each feature
            for feature_i in range(n_features):
                # for all values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # iterate through all unique values of feature column i and calculate the impurity
                for threshold in unique_values:
                    # split each feature value based on threshold where threshold is each unique value in feature_i column
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # calculate impurity
                        impurity = self.impurity_calculation(y, y1, y2)

                        # if curr impurity higher than previous largest_impurity (higher info gain), save new impurity and feature index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X of left subtree
                                "lefty": Xy1[:, n_features:],   # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]   # y of right subtree
                                }

        if largest_impurity > self.min_impurity:
            # build subtrees for the right and left branches since we're still above min impurity
            left_branch = self.build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            right_branch = self.build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], left_branch=left_branch, right_branch=right_branch)

        # once we are below min impurity, we are at a leaf node so calculate a value
        leaf_value = self.leaf_value_calculation(y)

        return DecisionNode(value=leaf_value, data=y)


    def predict_value(self, x, tree=None):
        # recursive search down tree until we reach a leaf node
        if tree is None:
            tree = self.root

        # if at leaf (there is a value) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # choose the feature that we will test
        feature_value = x[tree.feature_i]

        # determine if we will follow left or right branch
        if feature_value >= tree.threshold:
            branch = tree.left_branch
        else:
            branch = tree.right_branch

        # check subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        # predict value for each sample
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred
