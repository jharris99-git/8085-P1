import pandas as pd

pd.set_option('display.max_rows', 20)
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


# Import base CSV data from file.
base_data = pd.read_csv('./datasets/UNSW-NB15-BALANCED-TRAIN.csv', delimiter=',')


print(base_data)

final_prep(base_data)
