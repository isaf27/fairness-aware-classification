import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def get_transformer(cat_features, numeric_features):
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='error', sparse_output=False), cat_features),
        ('numeric', SimpleImputer(), numeric_features)
    ])


# ______________ Adult ______________
def read_adult(data_dir='adult'):
    colnames = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'income'
    ]
    train = pd.read_csv(Path(data_dir) / 'adult.data', header=None, names=colnames, skipinitialspace=True)
    test = pd.read_csv(Path(data_dir) / 'adult.test', header=None, names=colnames, skipinitialspace=True)
    return train, test


def get_adult(data_dir='adult'):
    train, test = read_adult(data_dir)
    
    is_protected_train = (train['sex'] == 'Female').astype('int')
    y_train = ((train['income'] == '>50K') | (train['income'] == '>50K.')).astype('int')
    train.drop(labels=['income'], axis=1, inplace=True)
    is_protected_test = (test['sex'] == 'Female').astype('int')
    y_test = ((test['income'] == '>50K') | (test['income'] == '>50K.')).astype('int')
    test.drop(labels=['income'], axis=1, inplace=True)
    
    cat_features = []
    numeric_features = []
    for name, tp in train.dtypes.items():
        if tp == 'object':
            cat_features.append(name)
        else:
            numeric_features.append(name)
    
    transformer = get_transformer(cat_features, numeric_features)
    X_train = transformer.fit_transform(train)
    X_test = transformer.transform(test)
    
    return X_train, np.array(y_train), np.array(is_protected_train), \
           X_test, np.array(y_test), np.array(is_protected_test)


# ______________ Bank ______________
def read_bank(data_dir='bank'):
    return pd.read_csv(Path(data_dir) / 'bank-full.csv', delimiter=';')


def get_bank(data_dir='bank', test_size=0.25, random_state=239):
    data = read_bank(data_dir)
    
    is_protected = (data['marital'] == 'married').astype('int')
    y = (data['y'] == 'yes').astype('int')
    data.drop(labels=['y'], axis=1, inplace=True)
    
    train, test, y_train, y_test, is_protected_train, is_protected_test = train_test_split(
        data, y, is_protected, test_size=test_size, random_state=random_state, stratify=(is_protected * 2 + y)
    )
    
    cat_features = []
    numeric_features = []
    for name, tp in train.dtypes.items():
        if tp == 'object':
            cat_features.append(name)
        else:
            numeric_features.append(name)
    
    transformer = get_transformer(cat_features, numeric_features)
    X_train = transformer.fit_transform(train)
    X_test = transformer.transform(test)
    
    return X_train, np.array(y_train), np.array(is_protected_train), \
           X_test, np.array(y_test), np.array(is_protected_test)


# ______________ Compass ______________
def read_compass(data_dir='compass'):
    return pd.read_csv(Path(data_dir) / 'propublicaCompassRecividism_data_fairml.csv' / 'propublica_data_for_fairml.csv')


# ______________ Census ______________
def read_census(data_dir='census-income'):
    colnames = [
        "age",
        "class of worker",
        "industry code",
        "occupation code",
        "adjusted gross income",
        "education",
        "wage per hour",
        "enrolled in edu inst last wk",
        "marital status",
        "major industry code",
        "major occupation code",
        "race",
        "hispanic Origin",
        "sex",
        "member of a labor union",
        "reason for unemployment",
        "full or part time employment stat",
        "capital gains",
        "capital losses",
        "divdends from stocks",
        "federal income tax liability",
        "tax filer status",
        "region of previous residence",
        "state of previous residence",
        "detailed household and family stat",
        "detailed household summary in household",
        "instance weight",
        "migration code-change in msa",
        "migration code-change in reg",
        "migration code-move within reg",
        "live in this house 1 year ago",
        "migration prev res in sunbelt",
        "num persons worked for employer",
        "family members under 18",
        "total person earnings",
        "country of birth father",
        "country of birth mother",
        "country of birth self",
        "citizenship",
        "total person income",
        "own business or self employed",
        "taxable income amount"
    ]
    train = pd.read_csv(Path(data_dir) / 'census-income.data', header=None, names=colnames, skipinitialspace=True)
    test = pd.read_csv(Path(data_dir) / 'census-income.test', header=None, names=colnames, skipinitialspace=True)
    return train, test
