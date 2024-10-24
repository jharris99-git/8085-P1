import argparse
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import classification_report

from prep import process_data
from train import feature_select, feature_sel_test_K

features = []

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
        # data = pd.read_csv('../test_data/' + args.testset, low_memory=False)
        data = pd.read_csv('../test_data/UNSW-NB15-BALANCED-TRAIN-HALVED.csv', low_memory=False)
        data = process_data(data)
        mdl_url = ''

        if args.model:
            mdl_url = '../models' + args.model
        else:
            match args.task:
                case 'Label':
                    match args.classifier:
                        case 'Label_RFC':
                            mdl_url = '../models/Label_RFC.pkl'
                            features = []
                        case 'Label_PCA':
                            mdl_url = '../models/label_PCA.pkl'
                            features = []
                        case 'label_m3':
                            mdl_url = '../models/label_m3.pkl'
                            features = []
                        case _:
                            print('invalid classifer model')
                            exit(2)
                case 'attack_cat':
                    match args.classifier:
                        case 'ac_m1':
                            mdl_url = '../models/ac_m1.pkl'
                            features = []
                        case 'attack_cat_PCA':
                            mdl_url = '../models/attack_cat_PCA.pkl'
                            factor = pd.factorize(data['attack_cat'])
                            data.attack_cat = factor[0]
                            features = []
                        case 'ac_m3':
                            mdl_url = '../models/ac_m3.pkl'
                            features = []
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
        # match args.task:
        #     case 'Label':
        #         data = feature_select(data, features)
        #         y_test = data['Label']
        #         x_test = data.drop('Label', axis=1)
        #     case 'attack_cat':
        #         if np.array(data.Label)[0] == 0:
        #             print('Label must be 1')
        #             exit(4)
        #         data = feature_select(data, features)
        #         y_test = data['attack_cat']
        #         x_test = data.drop('attack_cat', axis=1)

        y_pred = mdl.predict(feature_sel_test_K(data, args.task))
        print(np.array(y_pred))
        print(classification_report(y_true=data[args.task], y_pred=y_pred))

    except pd.errors as e:  # Hopefully works
        print('invalid test data')
        exit(1)

