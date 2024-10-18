import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import pickle


features = []


def feature_select_label(data: pd.DataFrame):
    # remove unnecessary features
    test_data = data[features]
    return test_data


def final_prep(data: pd.DataFrame):
    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)
    return label_data


def train_model(data: pd.DataFrame, model):
    y = data.Label
    x = data.drop('Label', axis=1)

    model = model.fit(x, y)

    kf = KFold(n_splits=5, shuffle=True, random_state=37)
    scores = cross_val_score(model, x, y, cv=kf, scoring='f1')
    return np.mean(scores)


def save_pkl(name: str, model):
    with open('../models/' + name + '.pkl', 'wb') as mdl_pkl:
        pickle.dump(model, mdl_pkl)

