import numpy as np
import progressbar


def mse(X):
    # calculate mean squared error 
    avg = np.average(X)
    errors = X - avg
    squared_error = errors**2
    return float(np.mean(squared_error))

def divide_on_feature(X, feature_i, threshold):
    # split features of feature_i index based on threshold
    split_func = lambda sample: sample[feature_i] >= threshold

    X_1 = [sample for sample in X if split_func(sample)]
    X_2 = [sample for sample in X if not split_func(sample)]
    X_1 = np.array(X_1)
    X_2 = np.array(X_2)

    return X_1, X_2

bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]

# loss functions: 
class squared_error(): 
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)
    
class huber():
    def __init__(self): pass

    def loss(self, y, y_pred):
        gamma = 1.0
        d = np.abs(y - y_pred)
        huber_mse = 0.5 * np.power((y - y_pred), 2)
        huber_mae = gamma * (d - 0.5 * gamma)
        return np.where(d <= gamma, huber_mse, huber_mae)