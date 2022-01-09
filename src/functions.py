import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score


def random_value_imputation(database, feature):
    """Function for imputing a random value in the missing or null values of a feature in the dataset
    :param
        database: complete database.
        feature: name of the feature to impute the values
    :return
        database: complete database with the imputed values"""
    random_sample = database[feature].dropna().sample(database[feature].isna().sum())
    random_sample.index = database[database[feature].isnull()].index
    database.loc[database[feature].isnull(), feature] = random_sample
    return database


def impute_mode(database, feature):
    """Function for imputing the mode in the missing or null values of a feature in the dataset
        :param
            database: complete database.
            feature: name of the feature to impute the values
        :return
            database: complete database with the imputed values"""
    mode = database[feature].mode()[0]
    database[feature] = database[feature].fillna(mode)
    return database


def get_num_people_by_age_category(database):
    """Generate a new feature from a database with the age group of each instance
    :param
        database: original database to add the feature
    :return
        database: new database with the new feature added"""
    database["AGE_GROUP"] = pd.cut(x=database['AGE'], bins=[0, 20, 40, 60, 80, 100, 120],
                                   labels=["0-20", "21-40", "41-60", '61-80', '81-100', '101-120'])
    return database


def generate_model(x_train, y_train, x_test, y_test, model):
    """Function for generating the machine learning model, fit the data and evaluate the results.
    :param
        x_train: database to be use for training the model
        y_train: label of the train dataset
        x_test: database to be use for evaluating the model
        y_test: label of the test dataset
        model: model to be use
    :returns
        model: model trained
        acc_train: accuracy of the training dataset
        acc_test: accuracy of the test dataset
        confusion_matrix_model: confusion matrix of the test set
        classification_report_model: classification report of the test set"""
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred = model.predict(x_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)
    confusion_matrix_model = confusion_matrix(y_test, y_pred)
    classification_report_model = classification_report(y_test, y_pred)
    return model, acc_train, acc_test, confusion_matrix_model, classification_report_model
