
import numpy as np

def calculate_rmse(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    return np.sqrt(((actual - predicted) ** 2).mean())
