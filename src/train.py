import pickle
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from prep import process_data

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 5000)

SKLClassifier = Union[RandomForestClassifier, MLPClassifier, KNeighborsClassifier, SVC, LogisticRegression]

true_class = []
pred_class = []

ac_features = []
label_features = []


def feature_sel_test_J(data: pd.DataFrame):
    # Used for aggregated classification report in KFold
    global true_class
    global pred_class

    # Split data for label classification
    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)

    # Split data for attack category classification
    ac_data = pd.DataFrame(data)
    ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
    ac_data = ac_data.drop('Label', axis=1)

    # ~~~~~~~ Label ~~~~~~~ #
    label_data_y = label_data['Label']
    label_data_X = label_data.drop('Label', axis=1)

    # Scale data, not always necessary.
    # scaler = StandardScaler().fit(label_data_X)
    # label_data_X = scaler.transform(label_data_X)

    # Create RFC for use in a SelectFromModel feature selector and fit to determine column importance.
    est = RandomForestClassifier(n_estimators=1000, verbose=2, n_jobs=10)
    sel = SelectFromModel(est)

    sel = sel.fit(label_data_X, label_data_y)

    # Get selected labels according to SelectFromModel and add them to the columns for use in the final model
    sel_feat_ind_Label = np.where(sel.get_support())[0]
    sel_feat_Label = label_data.columns[sel_feat_ind_Label]

    print(sel_feat_Label)

    sel_label_data_cols = sel_feat_Label.to_list() + ['Label']
    sel_label_data = label_data[sel_label_data_cols]

    # Define model for kfold using selected features
    model = RandomForestClassifier(n_estimators=300, verbose=2, n_jobs=10)
    kfold_means = train_score_model('Label', sel_label_data, model)

    # If the mean f1 score of kfold tests > 0.95, fit the model with more estimators and save the binary.
    if kfold_means > 0.95:
        y = sel_label_data['Label']
        x = sel_label_data.drop('Label', axis=1)

        # Fit final model.
        model_fin = RandomForestClassifier(n_estimators=2000, verbose=2, n_jobs=10)
        model_fin.fit(x, y)
        # Pickle and save model as binary.
        save_pkl('Label_RFC', model_fin)

    # Print classification report of aggregated predictions.
    print(classification_report(y_true=true_class, y_pred=pred_class))

    # Clear aggregated values.

    true_class = []
    pred_class = []

    # ~~~~~~~ Attack Category ~~~~~~~ #

    return sel_feat_Label


def feature_sel_test_K(data: pd.DataFrame):
    return data


def feature_sel_test_L(data: pd.DataFrame):
    return data


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


def aggregating_f1_scorer(y_true, y_pred):
    true_class.extend(y_true)
    pred_class.extend(y_pred)
    return f1_score(y_true, y_pred)


def train_score_model(target_col: str, data: pd.DataFrame, model: SKLClassifier) -> (str, SKLClassifier):
    """

    :param target_col:
    :param data:
    :param model:
    :return:
    """
    y = data[target_col]
    x = data.drop(target_col, axis=1)

    # model = model.fit(x, y)

    kf = KFold(n_splits=5, shuffle=True, random_state=37)
    scores = cross_val_score(model, x, y, cv=kf, scoring=make_scorer(aggregating_f1_scorer))
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

NAME = 'J'

match NAME:
    case 'J':
        feature_sel_test_J(base_data)
    case 'K':
        feature_sel_test_K(base_data)
    case 'L':
        feature_sel_test_L(base_data)


