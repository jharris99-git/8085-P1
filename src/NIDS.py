import argparse
import numpy as np
import pandas as pd
import pickle
from train import feature_select

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
        data = pd.read_csv('../test_data/' + args.testset, low_memory=False)

        mdl_url = ''

        if args.model:
            mdl_url = '../models' + args.model
        else:
            match args.task:
                case 'Label':
                    match args.classifier:
                        case 'Label_RFC':
                            mdl_url = '../models/Label_RFC.pkl'
                            features = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'smeansz', 'dmeansz', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'state_CLO', 'state_FIN']
                        case 'label_m2':
                            mdl_url = '../models/label_m2.pkl'
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
                            mdl_url = '../models/attack_cat_RFC.pkl'
                            features = ['dur', 'sbytes', 'dbytes', 'sttl', 'sloss', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'ct_flw_http_mthd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'proto_ttp', 'state_URN', 'service_dhcp', 'service_ftp-data']
                        case 'ac_m2':
                            mdl_url = '../models/ac_m2.pkl'
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
        match args.task:
            case 'Label':
                y_test = data['Label']
                x_test = data[features]
            case 'attack_cat':
                if data['Label'].isin(0):
                    print('Label must be 1')
                    exit(4)
                data = feature_select(data, features)
                y_test = data['attack_cat']
                x_test = data[features]

        y_pred = mdl.predict(x_test, y_test)
        print('Predicted values: ' + np.array(y_pred))

    except pd.errors as e:  # Hopefully works
        print('invalid test data')
        exit(1)

