import util
import pandas as pd
import numpy as np
from typing import Union, Tuple
from queue import Queue
import math


def mse_split(vals_array: np.ndarray, indices: list[int]) -> float:
    array = util.get_1d(vals_array)
    if array.shape[0] == 0:
        return 0

    subarray = array[indices]
    avg_val = np.average(subarray)

    squared_error = (subarray - avg_val) ** 2

    mean_squared_error = np.average(squared_error)

    return float(mean_squared_error)


def ssr_split(vals_array: np.ndarray) -> float:
    array = util.get_1d(vals_array)
    if array.shape[0] == 0:
        return 0

    avg_val = np.average(array)

    squared_residual = (array - avg_val) ** 2

    sum_squared_residual = np.sum(squared_residual)

    return float(sum_squared_residual)


class DecisionTree:
    def __init__(
        self,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int | None = None,
        max_features: int | float | str | None = None,
    ):
        self.root = None
        self.config = {
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_impurity_decrease": min_impurity_decrease,
        }
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y) -> None:
        bfs_queue = Queue()
        y = util.get_1d(y)
        node_num = 0
        self.root = DecisionTreeNode(np.arange(y.shape[0]), X, y, self.config, node_num)
        bfs_queue.put(self.root)
        while not bfs_queue.empty():
            curr_node = bfs_queue.get()
            if curr_node:
                left, right = curr_node.create_children()
                if left:
                    node_num += 1
                    left.node_num = node_num
                    bfs_queue.put(left)
                if right:
                    node_num += 1
                    right.node_num = node_num
                    bfs_queue.put(right)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        evaluate_row = lambda row: self.evaluate_point(row)
        ret = np.apply_along_axis(self.evaluate_point, axis=1, arr=X)
        return ret

    def evaluate_point(self, data_point: np.ndarray) -> float:
        if not self.root:
            raise ValueError("The model has not been trained yet.")
        else:
            return self.root.evaluate(data_point)

    def json(self) -> dict:
        if not self.root:
            return {"Result": "No root - hasn't been fitted yet"}
        return self.root.json()


class DecisionTreeNode:
    def __init__(
        self,
        indices: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        config: dict,
        node_num: int | None = None,
    ):
        self.indices = indices
        self.X = X
        self.y = y
        self.left_child = None
        self.right_child = None
        self.ssr = None
        self.col_num = None
        self.split_value = None
        self.average = float(np.average(y[indices]))
        self.config = config
        self.node_num = node_num

    def json(self) -> dict:
        ret = {
            "col_num": self.col_num,
            "split_value": self.split_value,
            "avg": self.average,
            "indices": list(int(i) for i in self.indices),
            "node_num": self.node_num,
        }
        if self.left_child:
            ret["left_num"] = self.left_child.node_num
        if self.right_child:
            ret["right_num"] = self.right_child.node_num
        if self.left_child:
            ret["left"] = self.left_child.json()
        if self.right_child:
            ret["right"] = self.right_child.json()
        return ret

    def evaluate(self, data_point: np.ndarray) -> float:
        # data = util.get_1d(data_point)

        value_to_check = data_point[self.col_num]
        if self.left_child or self.right_child:
            if not self.left_child or not self.right_child:
                raise Exception("Node only has one child.")
            elif value_to_check < self.split_value:
                return self.left_child.evaluate(data_point)
            elif value_to_check >= self.split_value:
                return self.right_child.evaluate(data_point)
            else:
                raise ValueError("Could not create prediction")
        else:
            return self.average

    def create_children(
        self,
    ) -> Tuple[Union["DecisionTreeNode", None], Union["DecisionTreeNode", None]]:
        # min_samples_split: int
        if isinstance(self.config["min_samples_split"], int):
            min_samples_split = self.config["min_samples_split"]
        elif isinstance(self.config["min_samples_split"], float):
            min_samples_split = math.ceil(
                self.config["min_samples_split"] * self.X.shape[0]
            )
        else:
            raise TypeError("min_samples_split needs to be float or an int")
        if len(list(self.indices)) < min_samples_split:
            return None, None

        left, right = self.get_split_nodes()
        # min_samples_leaf: int
        if isinstance(self.config["min_samples_leaf"], int):
            min_samples_leaf = self.config["min_samples_leaf"]
        elif isinstance(self.config["min_samples_leaf"], float):
            min_samples_leaf = math.ceil(
                self.config["min_samples_leaf"] * self.X.shape[0]
            )
        else:
            raise TypeError("min_samples_leaf needs to be a float or an int")

        if (
            len(list(left.indices)) < self.config["min_samples_leaf"]
            or len(list(right.indices)) < self.config["min_samples_leaf"]
        ):
            return None, None

        # min_impurity_decrease taken from sklearn docs
        curr_impurity = self.impurity()
        left_impurity = left.impurity()
        right_impurity = right.impurity()
        n = self.X.shape[0]
        n_t = self.indices.shape[0]
        n_t_r = right.indices.shape[0]
        n_t_l = left.indices.shape[0]
        impurity_decrease = (
            n_t
            / n
            * (
                curr_impurity
                - n_t_r / n_t * right_impurity
                - n_t_l / n_t * left_impurity
            )
        )
        if impurity_decrease < self.config["min_impurity_decrease"]:
            return None, None

        self.left_child = left
        self.right_child = right
        return self.left_child, self.right_child

    def impurity(self) -> float:
        avg = np.average(self.y)
        errors = self.y - avg
        squared_error = errors**2
        return float(np.mean(squared_error))

    def get_split_nodes(self) -> Tuple["DecisionTreeNode", "DecisionTreeNode"]:
        # split = self.get_best_split(self.X[self.indices], self.y[self.indices])
        split = self.get_best_split(self.X, self.y)
        left_idx = np.where(self.X[:, split["col_num"]] < split["split_value"])
        right_idx = np.where(self.X[:, split["col_num"]] >= split["split_value"])

        left_idx = np.intersect1d(left_idx, np.array(self.indices))
        right_idx = np.intersect1d(right_idx, np.array(self.indices))
        # print(left_idx, right_idx, sep="\n")

        return (
            DecisionTreeNode(left_idx, self.X, self.y, self.config),
            DecisionTreeNode(right_idx, self.X, self.y, self.config),
        )

    def get_best_split(self, feature_array: np.ndarray, label_array: np.ndarray):
        ssr_splits_per_feature = []
        for i in range(feature_array.shape[1]):
            ith_feature_split = self.best_split_feature(
                # feature_array[:, i], label_array
                feature_array[self.indices, i],
                label_array[self.indices],
            )
            ith_feature_split["col_num"] = i
            ssr_splits_per_feature.append(ith_feature_split)
        sorted_list = sorted(ssr_splits_per_feature, key=lambda x: x["ssr"])
        best_split = sorted_list[0]
        # print(sorted_list)
        self.ssr = best_split["ssr"]
        self.col_num = best_split["col_num"]
        self.split_value = best_split["split_value"]
        return best_split

    def best_split_feature(self, feature_array: np.ndarray, label_array: np.ndarray):
        print(feature_array.shape)
        array_to_split = np.column_stack(
            (util.get_1d(feature_array), util.get_1d(label_array))
        )
        sorted_array = np.sort(array_to_split, axis=0)

        split_candidates = self.get_split_candidates(sorted_array[:, 0])
        min_candidate_value = None
        min_candidate_ssr = np.inf
        for split_value in split_candidates:
            index = np.searchsorted(sorted_array[:, 0], split_value)

            left_array = sorted_array[:index, :]
            right_array = sorted_array[index:, :]

            candidate_ssr = self.get_candidate_ssr(left_array, right_array)

            if min_candidate_ssr > candidate_ssr:
                min_candidate_value = split_value
                min_candidate_ssr = candidate_ssr
        return {"ssr": min_candidate_ssr, "split_value": min_candidate_value}

    def get_split_candidates(self, feature_array: np.ndarray):
        return (feature_array[1:] + feature_array[:-1]) / 2.0

    def get_candidate_ssr(
        self, left_array: np.ndarray, right_array: np.ndarray
    ) -> float:
        left_ssr = ssr_split(left_array[:, 1])
        right_ssr = ssr_split(right_array[:, 1])

        return left_ssr + right_ssr
