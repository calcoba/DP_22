import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import shap

from src.functions import *

# Create a directory for storing the images created during the execution of the program
try:
    os.mkdir('images')
    print('Directory images created')
except FileExistsError:
    print('Directory images already exist')

# Ask the user for the name of the file that must be placed in the data directory and load it
data_loaded = False
while not data_loaded:
    working_data = input('Please enter the name of the file with the data: ')
    try:
        data = pd.read_csv('data/' + working_data)
        data_loaded = True
    except FileNotFoundError:
        print('Please enter a valid file name')

# Drop the ID column, as it does not contain any useful information
data = data.drop('ID', axis=1)

# Data
# Create a list with the features names where 0 is considered as a null value and replace it in the dataset
nan_column = ['AGE', 'TEMP', 'HEART_RATE', 'GLUCOSE', 'SAT_O2', 'BLOOD_PRES_SYS', 'BLOOD_PRES_DIAS']
data[nan_column] = data[nan_column].replace(0, np.nan)
# Drop a feature is more than 30% of the data available are null values
perc = 30.0
min_count = int(((100 - perc) / 100) * data.shape[0] + 1)
data = data.dropna(axis=1, thresh=min_count)

# For the remaining features impute the null values with two options, depending on their amount. With mode or random
for col in data.columns:
    if data[col].isna().sum() <= 0.1 * data.shape[0]:
        data = impute_mode(data, col)
    else:
        data = random_value_imputation(data, col)

# Eliminate outliers that are clearly a wrong input value
data = data[data.AGE < 120]
data = data[data.TEMP > 0.1]
data = data[data.HEART_RATE < 250]
data = data[data.SAT_O2 > 30]

# Save different plots for the exploratory data analysis in the images directory
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


# Extracting categorical and numerical columns
cat_cols = [col for col in data.columns if data[col].dtype == 'object']
num_cols = [col for col in data.columns if data[col].dtype != 'object']

# Create a label encoder and One hot encoder
le = LabelEncoder()
ohe = OneHotEncoder()

# Depending on the number of different values in the categorical columns select the label encoder or the one hot
# encoder.
for col in cat_cols:
    if data[col].nunique() == 2:
        data[col] = le.fit_transform(data[col])
    else:
        data[col] = ohe.fit_transform(data[col])

# Save a correlation plot between the different features in the images directory
plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linecolor='lightgrey', )
plt.savefig('images/corr.png')
plt.close()
print('Correlation image created!')

# Generate a new column in the dataset with the age group of the instance
covid = get_num_people_by_age_category(data)

# Generate the survival function based on the age group and save the result in the images directory
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
# The DAYS_HOSPITAL and DAYS_ICU features are dropped because it is data that is not available when the patient enter to
# the hospital
x = covid.drop(columns=['EXITUS', 'AGE_GROUP', 'DAYS_HOSPITAL', 'DAYS_ICU'], axis=1)

# Split the dataset in train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

# Perform a feature selection based on a chi2 test. P value must be smaller than 0.05 in order to select the feature
chi2_test = chi2(x_train, y_train)
unusefull_columns = []
for p_value, column_name in zip(chi2_test[1], x.columns):
    if p_value > 0.05:
        unusefull_columns.append(column_name)
print(unusefull_columns)

# Save a plot with the value of the chi2 statistic for each column in the images directory
df = pd.DataFrame({'Nombre_feat': x.columns, 'values': chi2_test[0]})
df.plot.barh(x='Nombre_feat', y='values', rot=0, figsize=(10, 10))
plt.savefig('images/chicuadrado.png')
plt.close()
print('Chi2 image created!')
print('--------------')
print('--------------')

# Normalizing data
x_train = x_train.drop(columns=unusefull_columns)
x_test = x_test.drop(columns=unusefull_columns)
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

# Generate the and evaluate the different model used (Decision Tree, K Neighbors, Neural Network and Random Forest
# Display a summary with the evaluation of each of the models in the command line
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
print('--------------')

mlp_model = generate_model(X_train, y_train, X_test, y_test, MLPClassifier(max_iter=1000, random_state=0))

print(f"Training Accuracy of Neural Network is {mlp_model[1].round(2)}")
print(f"Test Accuracy of Neural Network is {mlp_model[2].round(2)} \n")

print(f"Confusion Matrix :- \n{mlp_model[3]}\n")
print(f"Classification Report :- \n {mlp_model[4]}")
print('--------------')

lr_model = generate_model(X_train, y_train, X_test, y_test, LogisticRegression(random_state=0))

print(f"Training Accuracy of Logistic Regression is {lr_model[1].round(2)}")
print(f"Test Accuracy of Logistic Regression is {lr_model[2].round(2)} \n")

print(f"Confusion Matrix :- \n{lr_model[3]}\n")
print(f"Classification Report :- \n {lr_model[4]}")
print('--------------')

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
print('--------------')
print('--------------')

# Save a explanatory plot for each of the models in the images directory
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

shap.summary_plot(shap_values_dt[0], features=x_test,
                  feature_names=x_test.columns, plot_size=(15, 8), show=False, plot_type='dot')
plt.savefig("images/explained_model_dt.png")
plt.close()

shap.summary_plot(shap_values_lg, features=x_test,
                  feature_names=x_test.columns, plot_size=(15, 8), show=False, plot_type='dot')
plt.savefig("images/explained_model_lg.png")
plt.close()

shap.summary_plot(shap_values_mlp, features=x_test,
                  feature_names=x_test.columns, plot_size=(15, 8), show=False, plot_type='dot')
plt.savefig("images/explained_model_mlp.png")
plt.close()

shap.summary_plot(shap_values_rf[0], features=x_test,
                  feature_names=x_test.columns, plot_size=(15, 8), show=False, plot_type='dot')
plt.savefig("images/explained_model_rf.png")
plt.close()

print('Explained model image created!')

models = np.array([dt_model, knn_model, mlp_model, rf_model], dtype=object)
best_model = np.argmax(models[:, 2], axis=0)
if best_model == 3:
    models[3][0] = rf_model[0].best_estimator_
print('--------------')
print()
print('The best model is:')
print(models[best_model][0])

# Save a confusion matrix in an image format for the best model in the image directory
plt.figure(figsize=(5, 5))
sns.heatmap(models[best_model][3], annot=True, square=True, fmt='g')
plt.savefig("images/confusion_matrix_best_model.jpg")
plt.close()
print('Confusion matrix image created!')
print('--------------')

# Select the best model for allow predictions from the command line
final_model = models[best_model][0]
feature_names = x_train.columns
if best_model == 1 or best_model == 2:
    x_train, x_test = X_train, X_test
final_model.fit(x_train, y_train)
print()
print()
while True:
    print('''***************************************
*                                     *
*       Please select an option       *
*                                     *
*  1. Make prediction with best model *
*  2. Exit the program                *
*                                     *
***************************************''')
    try:
        option = int(input('Selection: '))
    except ValueError:
        print('Please enter a valid option')
        print('--------------')
        option = 0
    if option == 1:
        try:
            print('Please enter the instance features ({}):'.format(feature_names))
            features = input('Enter the features values separated by comas: ')
            features = list(features.split(','))
            features = np.array(features, dtype=float).reshape(1, -1)
            if best_model == 1 or best_model == 2:
                features = scaler.transform(features)
            instance_pred = final_model.predict(features)
            print()
            print('Prediction for the instance: {}'.format(instance_pred))
            print('--------------')
        except:
            print('Please enter a valid option')
            print('--------------')
    elif option == 2:
        sys.exit()
    else:
        print('Please enter a valid option')
        print('--------------')
