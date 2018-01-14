import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


path = os.getcwd() + '\data\stock_data.csv'
data = pd.read_csv(path, header = None, names=['Population', 'Profit'])
#print(data.head())
print(data.describe())
data.plot(kind = 'scatter', x='Population', y='Profit', figsize=(12,8))