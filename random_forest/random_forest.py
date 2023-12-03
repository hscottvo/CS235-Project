import numpy as np
import pandas as pd

import util
from decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int | None = None,
    ):
        self.config = {
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_impurity_decrease": min_impurity_decrease,
        }
        if random_state:
            np.random.seed(random_state)
