import numpy as np
import pandas as pd
import logging


def get_values(val: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(val, np.ndarray):
        return val
    elif isinstance(val, (pd.DataFrame, pd.Series)):
        return val.values
    else:
        error = (
            "Input must be a NumPy ndarray or a Pandas Series/DataFrame. Got {}".format(
                type(val)
            )
        )
        logging.error(error)
        raise TypeError(error)


def get_1d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    else:
        error = "Array must be shape (x) or (x, 1). Got shape {}".format(arr.shape)
        logging.error(error)
        raise ValueError(error)
