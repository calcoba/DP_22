

def random_value_imputation(database, feature):
    random_sample = database[feature].dropna().sample(database[feature].isna().sum())
    random_sample.index = database[database[feature].isnull()].index
    database.loc[database[feature].isnull(), feature] = random_sample
    return database


def impute_mode(database, feature):
    mode = database[feature].mode()[0]
    database[feature] = database[feature].fillna(mode)
    return database