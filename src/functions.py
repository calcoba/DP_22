import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

def random_value_imputation(database, feature):
    random_sample = database[feature].dropna().sample(database[feature].isna().sum())
    random_sample.index = database[database[feature].isnull()].index
    database.loc[database[feature].isnull(), feature] = random_sample
    return database


def impute_mode(database, feature):
    mode = database[feature].mode()[0]
    database[feature] = database[feature].fillna(mode)
    return database


def get_num_people_by_age_category(df):
    df["AGE_GROUP"] = pd.cut(x=df['AGE'], bins=[0, 20, 40, 60, 80, 100, 120],
                             labels=["0-20", "21-40", "41-60", '61-80', '81-100', '101-120'])
    return df


def generate_model(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred = model.predict(x_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)
    confusion_matrix_model = confusion_matrix(y_test, y_pred)
    classification_report_model = classification_report(y_test, y_pred)
    return model, acc_train, acc_test, confusion_matrix_model, classification_report_model
