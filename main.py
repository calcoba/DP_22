import pandas as pd
import numpy as np
from src.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import kaplanmeier as km
import os
try:
    os.mkdir('images')
    print('Directory images created')
except FileExistsError:
    print('Directory images already exist')

data_loaded = False
while not data_loaded:
    working_data = input('Please enter the name of the file with the data: ')
    try:
        data = pd.read_csv('data/'+working_data)
        data_loaded = True
    except FileNotFoundError:
        print('Please enter a valid file name')

data = data.drop('ID', axis=1)

# Data preprocessing

# esta parte no se me ocurre del como generalizarla
nan_column = ['AGE', 'TEMP', 'HEART_RATE', 'GLUCOSE', 'SAT_O2', 'BLOOD_PRES_SYS', 'BLOOD_PRES_DIAS']
data[nan_column] = data[nan_column].replace(0, np.nan)

perc = 30.0
min_count = int(((100-perc)/100)*data.shape[0] + 1)
data = data.dropna(axis=1, thresh=min_count)


for col in data.columns:
    if data[col].isna().sum() <= 0.1 * data.shape[0]:
        clean_data = impute_mode(data, col)
    else:
        random_value_imputation(data, col)

# checking for null values

print(data.isna().sum().sort_values(ascending=False))
data = data[data.AGE < 120]
data = data[data.TEMP > 0.1]
data = data[data.HEART_RATE < 250]
data = data[data.SAT_O2 > 30]
data.plot(kind='box', subplots=True, figsize=(20, 10))
plt.savefig('images/box_plot')

data.hist(figsize=(15, 15), bins=20)
plt.savefig('images/hist_plot')

sns.pairplot(data, hue='EXITUS')
plt.savefig('images/pairplot_plot')
