import pickle
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 5000)

SKLClassifier = Union[RandomForestClassifier, MLPClassifier, KNeighborsClassifier, SVC]

ac_features = []
label_features = []


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

