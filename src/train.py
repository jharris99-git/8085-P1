import pickle
from ast import literal_eval
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

from prep import process_data

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 5000)

SKLClassifier = Union[RandomForestClassifier, MLPClassifier, KNeighborsClassifier, SVC, LogisticRegression]

ac_features = []
label_features = []


def feature_sel_test_J(data: pd.DataFrame):
    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)

    print(label_data)

    ac_data = pd.DataFrame(data)
    ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
    ac_data = ac_data.drop('Label', axis=1)

    est = LogisticRegression()
    sel = RFECV(est, step=1, cv=5)

    print(label_data['ct_ftp_cmd'])

    label_data_y = label_data['Label']
    label_data_X = label_data.drop('Label', axis=1)

    sel = sel.fit(label_data_X, label_data_y)

    print(sel.ranking_)

    return data


def feature_sel_test_K(data: pd.DataFrame):
    return data


def feature_sel_test_L(data: pd.DataFrame, target: str):
    # Drop the other target column (whichever is not selected)
    if target == 'attack_cat':
        ac_data = pd.DataFrame(data)
        ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
        ac_data = ac_data.drop(ac_data[ac_data['attack_cat'] == 0].index, axis=0)
        final_data = ac_data.drop('Label', axis=1)
    else:
        label_data = pd.DataFrame(data)
        final_data = label_data.drop('attack_cat', axis=1)

    # Separate the features (x) and the selected target (y)
    x = final_data  # Features (excluding the target labels)
    y = final_data[target]  # Target variable (either 'attack_cat' or 'Label')

    # Convert categorical features to numerical using LabelEncoder
    label_encoder = LabelEncoder()
    for column in x.select_dtypes(include=['object', 'category']).columns:
        x[column] = label_encoder.fit_transform(x[column])

    # Apply the chi-square test
    chi2_scores, p_values = chi2(x, y)

    # Create a DataFrame with the results
    chi2_results = pd.DataFrame({'Feature': x.columns, 'Chi2 Score': chi2_scores, 'p-value': p_values})

    # Sort by p-value for analysis
    chi2_results = chi2_results.sort_values(by='p-value')

    print(f"Chi-Square Test Results ({target}):")
    print(f"{chi2_results}\n")

    return chi2_results


def feature_select(data: pd.DataFrame, features: list):
    """

    :param data:
    :param features:
    :return:
    """
    # remove unnecessary features
    test_data = data[features]
    return test_data


def initial_prep_attack_cat(data: pd.DataFrame):
    """

    :param data:
    :return:
    """
    ac_data = pd.DataFrame(data)
    ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
    ac_data = ac_data.drop('Label', axis=1)
    return ac_data


def initial_prep_label(data: pd.DataFrame):
    """

    :param data:
    :return:
    """
    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)
    return label_data


def train_score_model(target_col: str, data: pd.DataFrame, model: SKLClassifier) -> (str, SKLClassifier):
    """

    :param target_col:
    :param data:
    :param model:
    :return:
    """
    y = data[target_col]
    x = data.drop(target_col, axis=1)

    model = model.fit(x, y)

    kf = KFold(n_splits=5, shuffle=True, random_state=37)
    scores = cross_val_score(model, x, y, cv=kf, scoring='f1')
    return np.mean(scores)


def save_pkl(name: str, model: SKLClassifier):
    """

    :param name:
    :param model:
    :return:
    """
    with open('../models/' + name + '.pkl', 'wb') as mdl_pkl:
        pickle.dump(model, mdl_pkl)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


base_data = pd.read_csv('../datasets/UNSW-NB15-BALANCED-TRAIN.csv',
                        low_memory=False)


base_data = process_data(base_data)
generic_cats = base_data[base_data['attack_cat'] == 'Generic']
print(f'count: {len(generic_cats)}')

print(base_data)

NAME = 'L'

match NAME:
    case 'J':
        feature_sel_test_J(base_data)
    case 'K':
        feature_sel_test_K(base_data)
    case 'L':
        # feature_sel_test_L(base_data, 'Label') # seems to be working fine
        feature_sel_test_L(base_data, 'attack_cat')
