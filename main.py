import os

import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import shap

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

data = data[data.AGE < 120]
data = data[data.TEMP > 0.1]
data = data[data.HEART_RATE < 250]
data = data[data.SAT_O2 > 30]
data.plot(kind='box', subplots=True, figsize=(20, 10))
plt.savefig('images/box_plot')
plt.close()
print('Box plot image created!')

data.hist(figsize=(15, 15), bins=20)
plt.savefig('images/hist_plot')
plt.close()
print('Histogram image created!')

sns.pairplot(data, hue='EXITUS')
plt.savefig('images/pairplot_plot')
plt.close()
print('Pair plot image created!')

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
print('Correlation image created!')

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
plt.close()
print('Kaplan-Meier image created!')

# Generating a model

y = covid.EXITUS
x = covid.drop(columns=['EXITUS', 'AGE_GROUP', 'DAYS_HOSPITAL', 'DAYS_ICU'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

chi2_test = chi2(x_train, y_train)
unusefull_columns = []
for p_value, column_name in zip(chi2_test[1], x.columns):
    if p_value > 0.1:
        unusefull_columns.append(column_name)
print(unusefull_columns)

df = pd.DataFrame({'Nombre_feat': x.columns, 'values': chi2_test[0]})
df.plot.barh(x='Nombre_feat', y='values', rot=0, figsize=(10, 10))
plt.savefig('images/chicuadrado.png')
plt.close()
print('Chi2 image created!')

# Normalizing data
x_train = x_train.drop(columns=unusefull_columns)
x_test = x_test.drop(columns=unusefull_columns)

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

dt_model = generate_model(x_train, y_train, x_test, y_test, DecisionTreeClassifier(random_state=5))

print(f"Training Accuracy of Decision Tree is {dt_model[1].round(2)}")
print(f"Test Accuracy of Decision Tree is {dt_model[2].round(2)} \n")

print(f"Confusion Matrix :- \n{dt_model[3]}\n")
print(f"Classification Report :- \n {dt_model[4]}")

knn_model = generate_model(X_train, y_train, X_test, y_test, KNeighborsClassifier())

print(f"Training Accuracy of KNeighbors is {knn_model[1].round(2)}")
print(f"Test Accuracy of KNeighbors is {knn_model[2].round(2)} \n")

print(f"Confusion Matrix :- \n{knn_model[3]}\n")
print(f"Classification Report :- \n {knn_model[4]}")

mlp_model = generate_model(X_train, y_train, X_test, y_test, MLPClassifier(max_iter=1000, random_state=0))

print(f"Training Accuracy of Neural Network is {mlp_model[1].round(2)}")
print(f"Test Accuracy of Neural Network is {mlp_model[2].round(2)} \n")

print(f"Confusion Matrix :- \n{mlp_model[3]}\n")
print(f"Classification Report :- \n {mlp_model[4]}")

lr_model = generate_model(X_train, y_train, X_test, y_test, LogisticRegression(random_state=0))

print(f"Training Accuracy of Logistic Regression is {lr_model[1].round(2)}")
print(f"Test Accuracy of Logistic Regression is {lr_model[2].round(2)} \n")

print(f"Confusion Matrix :- \n{lr_model[3]}\n")
print(f"Classification Report :- \n {lr_model[4]}")

RF1 = RandomForestClassifier(random_state=0)
# define the grid of values to search
grid = dict()
grid['max_samples'] = np.arange(0.1, 1.1, 0.2)
grid['n_estimators'] = [10, 50, 100]
grid['max_depth'] = np.arange(1, 8, 1)

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the grid search procedure
grid_search = GridSearchCV(estimator=RF1, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

rf_model = generate_model(x_train, y_train, x_test, y_test, grid_search)

print(f"Training Accuracy of Random Forest is {rf_model[1].round(2)}")
print(f"Test Accuracy of Random Forest is {rf_model[2].round(2)} \n")

print(f"Confusion Matrix :- \n{rf_model[3]}\n")
print(f"Classification Report :- \n {rf_model[4]}")

plt.figure(figsize=(5, 5))
sns.heatmap(rf_model[3], annot=True, square=True, fmt='g')
plt.savefig("images/RF_matrix.jpg")
plt.close()
print('Confusion matrix image created!')

dt_model[0].fit(x_train, y_train)
explainer_dt = shap.TreeExplainer(dt_model[0])
shap_values_dt = explainer_dt.shap_values(x_test)

#lr_model[0].fit(X_train, y_train)
explainer_lg = shap.LinearExplainer(lr_model[0], X_train)
shap_values_lg = explainer_lg.shap_values(X_test)

#mlp_model[0].fit(X_train, y_train)
explainer_mlp = shap.KernelExplainer(mlp_model[0].predict, X_train)
shap_values_mlp = explainer_mlp.shap_values(X_test)

rf_model_best = rf_model[0].best_estimator_
rf_model_best.fit(x_train, y_train)
explainer_rf = shap.TreeExplainer(rf_model_best)
shap_values_rf = explainer_rf.shap_values(x_test)

shap.summary_plot(shap_values_dt[1], features=x_test,
                  feature_names=x_test.columns, plot_size=(15, 8), show=False, plot_type='dot')
plt.savefig("images/explained_model.png")
plt.close()

shap.summary_plot(shap_values_lg, features=x_test,
                  feature_names=x_test.columns, plot_size=(15, 8), show=False, plot_type='dot')
plt.savefig("images/explained_model_1.png")
plt.close()

shap.summary_plot(shap_values_mlp, features=x_test,
                  feature_names=x_test.columns, plot_size=(15, 8), show=False, plot_type='dot')
plt.savefig("images/explained_model_2.png")
plt.close()

shap.summary_plot(shap_values_rf[1], features=x_test,
                  feature_names=x_test.columns, plot_size=(15, 8), show=False, plot_type='dot')
plt.savefig("images/explained_model_3.png")
plt.close()

print('Explained model image created!')
