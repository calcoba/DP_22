import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src.functions import *

try:
    os.mkdir('images')
    print('Directory images created')
except FileExistsError:
    print('Directory images already exist')

data_loaded = False
while not data_loaded:
    working_data = input('Please enter the name of the file with the data: ')
    try:
        data = pd.read_csv('data/' + working_data)
        data_loaded = True
    except FileNotFoundError:
        print('Please enter a valid file name')

data = data.drop('ID', axis=1)

# Data preprocessing

# esta parte no se me ocurre del como generalizarla
nan_column = ['AGE', 'TEMP', 'HEART_RATE', 'GLUCOSE', 'SAT_O2', 'BLOOD_PRES_SYS', 'BLOOD_PRES_DIAS']
data[nan_column] = data[nan_column].replace(0, np.nan)

perc = 30.0
min_count = int(((100 - perc) / 100) * data.shape[0] + 1)
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
plt.close()

data.hist(figsize=(15, 15), bins=20)
plt.savefig('images/hist_plot')
plt.close()

sns.pairplot(data, hue='EXITUS')
plt.savefig('images/pairplot_plot')
plt.close()

# Feature enconding

# Extracting categorical and numerical columns

cat_cols = [col for col in data.columns if data[col].dtype == 'object']
num_cols = [col for col in data.columns if data[col].dtype != 'object']

le = LabelEncoder()
ohe = OneHotEncoder()

for col in cat_cols:
    if data[col].nunique() == 2:
        data[col] = le.fit_transform(data[col])
    else:
        data[col] = ohe.fit_transform(data[col])

# heatmap of data

plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linecolor='lightgrey', )
plt.savefig('images/corr.png')
plt.close()

covid = get_num_people_by_age_category(data)

kmf_covid = KaplanMeierFitter()
kmf_covid.fit(data.DAYS_HOSPITAL[data['AGE_GROUP'] == "0-20"],
              data['EXITUS'][data['AGE_GROUP'] == "0-20"], label='0-20')
ax = kmf_covid.plot_survival_function()
kmf_covid.fit(data.DAYS_HOSPITAL[data['AGE_GROUP'] == "21-40"],
              data['EXITUS'][data['AGE_GROUP'] == "21-40"], label='21-40')
ax = kmf_covid.plot_survival_function(ax=ax)
kmf_covid.fit(data.DAYS_HOSPITAL[data['AGE_GROUP'] == "41-60"],
              data['EXITUS'][data['AGE_GROUP'] == "41-60"], label='41-60')
ax = kmf_covid.plot_survival_function(ax=ax)
kmf_covid.fit(data.DAYS_HOSPITAL[data['AGE_GROUP'] == "61-80"],
              data['EXITUS'][data['AGE_GROUP'] == "61-80"], label='61-80')
ax = kmf_covid.plot_survival_function(ax=ax)
kmf_covid.fit(data.DAYS_HOSPITAL[data['AGE_GROUP'] == "81-100"],
              data['EXITUS'][data['AGE_GROUP'] == "81-100"], label='81-100')
ax = kmf_covid.plot_survival_function(ax=ax)
kmf_covid.fit(data.DAYS_HOSPITAL[data['AGE_GROUP'] == "101-120"],
              data['EXITUS'][data['AGE_GROUP'] == "101-120"], label='101-120')
ax = kmf_covid.plot_survival_function(ax=ax)
ax.get_figure().savefig("images/km_graph.png")

# Generating a model

y = covid.EXITUS
x = covid.drop(columns=['EXITUS', 'AGE_GROUP'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


best_features_Chi2 = SelectKBest(score_func=chi2, k=7)
fit_Chi2 = best_features_Chi2.fit(x_train, y_train)

# gráfico de barras utilizando matplotlib
df = pd.DataFrame({'Nombre_feat': x.columns, 'values': fit_Chi2.scores_})
plt.figure(figsize=(12, 12))
ax = df.plot.barh(x='Nombre_feat', y='values', rot=0)
plt.savefig('images/chicuadrado.jpg')
plt.close()

# Normalizing data

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

dt_model = generate_model(x_train, y_train, x_test, y_test, DecisionTreeClassifier(random_state=4))

print(f"Training Accuracy of Decision Tree is {dt_model[1]}")
print(f"Test Accuracy of Decision Tree is {dt_model[2].round(2)} \n")

print(f"Confusion Matrix :- \n{dt_model[3]}\n")
print(f"Classification Report :- \n {dt_model[4]}")

knn_model = generate_model(X_train, y_train, X_test, y_test, KNeighborsClassifier())

print(f"Training Accuracy of KNeighbors is {knn_model[1]}")
print(f"Test Accuracy of KNeighbors is {knn_model[2].round(2)} \n")

print(f"Confusion Matrix :- \n{knn_model[3]}\n")
print(f"Classification Report :- \n {knn_model[4]}")

mlp_model = generate_model(X_train, y_train, X_test, y_test, MLPClassifier(max_iter=1000, random_state=0))

print(f"Training Accuracy of KNeighbors is {mlp_model[1]}")
print(f"Test Accuracy of KNeighbors is {mlp_model[2].round(2)} \n")

print(f"Confusion Matrix :- \n{mlp_model[3]}\n")
print(f"Classification Report :- \n {mlp_model[4]}")
