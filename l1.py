import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
path = os.getcwd() + '\data\stock_data.csv' 

data = pd.read_csv(path, header=None,names=['population','profit'])
print(data.describe())
print(data.head())
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8)) 

def computeCost(X, y, theta):  
	inner = np.power(((X * theta.T) - y), 2)
	return np.sum(inner) / (2 * len(X))

# append a ones column to the front of the data set
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]  