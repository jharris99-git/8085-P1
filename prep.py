from ast import literal_eval

import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 5000)


def final_prep(data):
    ac_data = pd.DataFrame(data)
    ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
    ac_data = ac_data.drop('Label', axis=1)
    ac_data.to_csv('./datasets/prepped_attack_cat_data.csv', index=True)

    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)
    label_data.to_csv('./datasets/prepped_label_data.csv', index=True)


expected_dtypes = {'srcip': 'str',
                   'sport': 'str',
                   'dstip': 'str',
                   'dsport': 'str',
                   }

# Import base CSV data from file.
base_data = pd.read_csv('./datasets/UNSW-NB15-BALANCED-TRAIN.csv', dtype=expected_dtypes,
                        low_memory=False)


# Fix '-' and '0x' entries in sport and dsport
base_data.sport = ['00' if x == "-" else x for x in base_data.sport.tolist()]
base_data.sport = [literal_eval(x) if type(x) == str and x[0:2] == "0x" else int(x) for x in base_data.sport.tolist()]

base_data.dsport = ['00' if x == "-" else x for x in base_data.dsport.tolist()]
base_data.dsport = [literal_eval(x) if type(x) == str and x[0:2] == "0x" else int(x) for x in base_data.sport.tolist()]

# Fix whitespace values in ct_ftp_cmd, to impute or prune later.
base_data.ct_ftp_cmd = [None if x == " " else int(x) for x in base_data.ct_ftp_cmd.tolist()]

# Do pruning, imputation, etc.

# Drop redundant columns for Label and attack_cat prediction training data.
final_prep(base_data)
