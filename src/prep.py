from ast import literal_eval

import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 5000)


def final_prep(data: pd.DataFrame):
    """

    :param data:
    """
    ac_data = pd.DataFrame(data)
    ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
    ac_data = ac_data.drop('Label', axis=1)
    ac_data.to_csv('../datasets/prepped_attack_cat_data.csv', index=False)  # Not necessary, just example

    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)
    label_data.to_csv('../datasets/prepped_label_data.csv', index=False)  # Not necessary, just example


def prune(data: pd.DataFrame):
    """

    :param data:
    :return: pd.DataFrame, pruned data
    """
    # Drop rows with more than 4 null values
    # data = data[data.isnull().sum(axis=1) < 4]

    # Drop IP columns from the dataset
    data = data.drop(['srcip', 'dstip', 'sport', 'dsport'], axis=1)

    return data


def encode(data: pd.DataFrame):
    """

    :param data:
    :return: pd.DataFrame, with encoded columns
    """
    data = pd.get_dummies(data, columns=['proto', 'state', 'service'])
    return data


def impute(data: pd.DataFrame):
    """

    :param data:
    :return: pd.DataFrame, imputed data
    """
    data = data.fillna(0)
    return data


def process_data(data: pd.DataFrame):
    # Fix '-' and '0x' entries in sport and dsport
    data.sport = ['00' if x == "-" else x for x in base_data.sport.tolist()]
    data.sport = [literal_eval(x) if type(x) == str and x[0:2] == "0x" else int(x) for x in
                  base_data.sport.tolist()]

    data.dsport = ['00' if x == "-" else x for x in base_data.dsport.tolist()]
    data.dsport = [literal_eval(x) if type(x) == str and x[0:2] == "0x" else int(x) for x in
                   base_data.sport.tolist()]

    # Fix whitespace values in ct_ftp_cmd, to impute or prune later.
    data.ct_ftp_cmd = [None if x == " " else float(x) for x in base_data.ct_ftp_cmd.tolist()]

    # Fix attack_category whitespace and plural errors
    data.attack_cat = data.attack_cat.str.replace(" ", "")
    data.attack_cat = data.attack_cat.str.replace("Backdoors", "Backdoor")

    data = prune(data)
    data = encode(data)
    data = impute(data)
    return data


# Define expected dtypes
expected_dtypes = {'srcip': 'str',
                   'sport': 'str',
                   'dstip': 'str',
                   'dsport': 'str',
                   }

# Import base CSV data from file.
base_data = pd.read_csv('../datasets/UNSW-NB15-BALANCED-TRAIN-HALVED.csv', dtype=expected_dtypes,
                        low_memory=False)


# Fix, Prune, encode, and impute.
process_data(base_data)

# Do this column dropping in train.py
# final_prep(base_data)
