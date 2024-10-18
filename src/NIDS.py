import argparse
import numpy as np
import pandas as pd
import pickle
from train_label import feature_select_label
from train_attack_cat import feature_select_attack_cat


parser = argparse.ArgumentParser(
    prog='NIDS',
    description='',
    epilog=''
)

parser.add_argument('testset')
parser.add_argument('classifier')
parser.add_argument('task')
parser.add_argument('-m', '--model', action='store_const')


if __name__ == '__main__':
    args = parser.parse_args()
    try:
        data = pd.read_csv('../test_data/' + args.testset, low_memory=False)

        mdl_url = ''

        if args.model:
            mdl_url = '../models' + args.model
        else:
            match args.task:
                case 'Label':
                    match args.classifier:
                        case 'label_m1':
                            mdl_url = '../models/label_m1.pkl'
                        case 'label_m2':
                            mdl_url = '../models/label_m2.pkl'
                        case 'label_m3':
                            mdl_url = '../models/label_m3.pkl'
                        case _:
                            print('invalid classifer model')
                            exit(2)
                case 'attack_cat':
                    match args.classifier:
                        case 'ac_m1':
                            mdl_url = '../models/ac_m1.pkl'
                        case 'ac_m2':
                            mdl_url = '../models/ac_m2.pkl'
                        case 'ac_m3':
                            mdl_url = '../models/ac_m3.pkl'
                        case _:
                            print('invalid classifer model')
                            exit(2)
                case _:
                    print('invalid task')
                    exit(3)

        with open(mdl_url, 'rb') as mdl_pkl:
            mdl = pickle.load(mdl_pkl)

        x_test = None
        y_test = None
        match args.task:
            case 'Label':
                data = feature_select_label(data)
                y_test = data['Label']
                x_test = data.drop('Label', axis=1)
            case 'attack_cat':
                data = feature_select_attack_cat(data)
                y_test = data['attack_cat']
                x_test = data.drop('attack_cat', axis=1)

        y_pred = mdl.predict(x_test, y_test)
        print('Predicted values: ' + np.array(y_pred))

    except pd.errors as e:
        print('invalid test data')
        exit(1)

