import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

path = os.getcwd() + '\data\stock_data.csv'

data = pd.read_csv(path, header=None, names=['population', 'profit'])
# data.describe()
data.head()


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
