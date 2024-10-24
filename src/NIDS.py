import gzip

import argparse
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import classification_report

from prep import process_data, prune
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
        data = pd.read_csv('./test_data/' + args.testset, low_memory=False, names=['srcip', 'sport', 'dstip', 'dsport',
                                                                                   'proto', 'state', 'dur', 'sbytes',
                                                                                   'dbytes', 'sttl', 'dttl', 'sloss',
                                                                                   'dloss', 'service', 'Sload', 'Dload',
                                                                                   'Spkts', 'Dpkts', 'swin', 'dwin',
                                                                                   'stcpb', 'dtcpb', 'smeansz',
                                                                                   'dmeansz', 'trans_depth',
                                                                                   'res_bdy_len', 'Sjit', 'Djit',
                                                                                   'Stime', 'Ltime', 'Sintpkt',
                                                                                   'Dintpkt', 'tcprtt', 'synack',
                                                                                   'ackdat', 'is_sm_ips_ports',
                                                                                   'ct_state_ttl', 'ct_flw_http_mthd',
                                                                                   'is_ftp_login', 'ct_ftp_cmd',
                                                                                   'ct_srv_src', 'ct_srv_dst',
                                                                                   'ct_dst_ltm', 'ct_src_ ltm',
                                                                                   'ct_src_dport_ltm',
                                                                                   'ct_dst_sport_ltm', 'ct_dst_src_ltm',
                                                                                   'attack_cat', 'Label'])

        # TODO: Data preparation

        data = prune(data)

        # TODO: Make dummies from categorical types for proto, service, and state.
        # DO NOT use process_data. attack_cat and Label do not exist in input test data

        mdl_url = ''

        if args.model:
            mdl_url = './models' + args.model
        else:
            match args.task:
                case 'Label':
                    match args.classifier:
                        case 'Label_RFC':
                            mdl_url = './models/Label_RFC.pkl'
                            features = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'smeansz', 'dmeansz', 'Sjit', 'Djit', 'Stime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'proto_tcf', 'state_CLO', 'state_FIN']
                        case 'Label_PCA':
                            mdl_url = '../models/label_PCA.pkl'
                            features = []
                        case 'label_m3':
                            mdl_url = './models/label_m3.pkl'
                            features = []
                        case _:
                            print('invalid classifer model')
                            exit(2)
                case 'attack_cat':
                    match args.classifier:
                        case 'attack_cat_RFC':
                            mdl_url = './models/attack_cat_RFC.pkl'
                            features = ['dur', 'sbytes', 'dbytes', 'sttl', 'sloss', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'ct_flw_http_mthd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'proto_ttp', 'state_URN', 'service_dhcp', 'service_ftp-data']
                        case 'attack_cat_PCA':
                            mdl_url = './models/attack_cat_PCA.pkl'
                            factor = pd.factorize(data['attack_cat'])
                            data.attack_cat = factor[0]
                            features = []
                        case 'ac_m3':
                            mdl_url = './models/ac_m3.pkl'
                            features = []
                        case _:
                            print('invalid classifer model')
                            exit(2)
                case _:
                    print('invalid task')
                    exit(3)

        mdl = None
        with gzip.open(mdl_url, 'rb') as mdl_pkl:
            mdl = pickle.load(mdl_pkl)

        x_test = None
        y_test = None
        match args.task:
            case 'Label':
                print(data)
                x_test = data[features]
            case 'attack_cat':
                x_test = data[features]


        y_pred = mdl.predict(x_test, y_test)
        print('Predicted values: ' + np.array(y_pred))

        # Can't use bc mystery test data won't have y_true
        # print(classification_report(y_true=data['attack_cat'], y_pred=y_pred))



    except pd.errors as e:  # Hopefully works
        print('invalid test data')
        exit(1)

