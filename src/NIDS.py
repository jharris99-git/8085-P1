import gzip

import argparse
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import classification_report

from prep import prune, proto_dtype, state_dtype, service_dtype, encode
from train import feature_select, feature_sel_test_K

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 205)
pd.set_option('display.width', 5000)

features = []

expected_dtypes = {'srcip': 'str',
                   'sport': 'str',
                   'dstip': 'str',
                   'dsport': 'str',
                   'proto': proto_dtype,
                   'state': state_dtype,
                   'service': service_dtype
                   }

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
        # Parse data, using pre-defined categorical dtypes for certain columns.
        data = pd.read_csv('./test_data/' + args.testset, low_memory=False, dtype=expected_dtypes,
                           names=['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
                                  'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
                                  'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
                                  'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
                                  'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
                                  'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm',
                                  'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'])

        # Prepare data
        data = prune(data)
        data = encode(data)

        mdl_url = ''

        # If --model is passed as an argument, select chosen model from ./models folder.
        if args.model:
            mdl_url = './models' + args.model
        else:
            match args.task:
                case 'Label':
                    # Label classifier and feature selector.
                    match args.classifier:
                        case 'Label_RFC':
                            mdl_url = './models/Label_RFC.pkl'
                            features = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'dwin', 'dtcpb', 'smeansz', 'dmeansz', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'state_CLO', 'state_FIN']
                        case 'Label_PCA':
                            mdl_url = '../models/Label_MLP.pkl'
                            features = []
                        case 'Label_CHI':
                            mdl_url = '../models/Label_CHI.pkl'
                            features = ['stcpb', 'dtcpb', 'Sload', 'Dload', 'dbytes', 'res_bdy_len', 'sbytes', 'Stime', 'Ltime', 'Djit', 'Sjit', 'dmeansz', 'sttl', 'Sintpkt', 'swin', 'dwin', 'Dpkts', 'Dintpkt', 'Spkts', 'dloss', 'ct_dst_src_ltm', 'ct_src_dport_ltm', 'ct_srv_dst', 'ct_srv_src', 'ct_dst_sport_ltm', 'dttl', 'smeansz', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_state_ttl', 'sloss', 'state_INT', 'proto_tcp', 'state_FIN', 'state_CON', 'service_dns', 'proto_udp', 'service_-', 'proto_unas', 'service_ftp-data', 'service_ssh', 'tcprtt', 'ct_flw_http_mthd', 'ct_ftp_cmd', 'ackdat', 'service_smtp', 'trans_depth', 'is_ftp_login', 'synack']
                        case _:
                            print('invalid classifer model')
                            exit(2)
                case 'attack_cat':
                    # Attack category classifier and feature selector.
                    match args.classifier:
                        case 'attack_cat_RFC':
                            mdl_url = './models/attack_cat_RFC.pkl'
                            features = ['dur', 'sbytes', 'dbytes', 'sttl', 'sloss', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'ct_flw_http_mthd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'proto_ttp', 'state_URN', 'service_dhcp', 'service_ftp-data']
                        case 'attack_cat_MLP':
                            mdl_url = '../models/attack_cat_MLP.pkl'
                            factor = pd.factorize(data['attack_cat'])
                            data.attack_cat = factor[0]
                            features = []
                        case 'attack_cat_CHI':
                            mdl_url = '../models/attack_cat_CHI.pkl'
                            features = ['dtcpb', 'stcpb', 'Sload', 'Dload', 'sbytes', 'Sjit', 'dbytes', 'res_bdy_len', 'Djit', 'Sintpkt', 'Dintpkt', 'dmeansz', 'swin', 'dwin', 'dttl', 'smeansz', 'Spkts', 'Dpkts', 'Stime', 'Ltime', 'sloss', 'ct_dst_src_ltm', 'ct_srv_dst', 'ct_srv_src', 'dloss', 'ct_src_dport_ltm', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_dst_sport_ltm', 'sttl', 'dur', 'service_-', 'proto_tcp', 'state_FIN', 'service_dns']
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
        cats = None

        # Cut features and enumerate possible results.
        match args.task:
            case 'Label':
                x_test = data[features]
                cats = [0, 1]
            case 'attack_cat':
                x_test = data[features]
                cats = ['Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance',
                        'Backdoor', 'Analysis', 'Shellcode', 'Worms']

        # Predict using the input data rows.
        y_pred = mdl.predict(x_test)

        # Force predictions to int and convert from enumerated value (could have been one line, no time to test).
        y_pred = [int(val) for val in y_pred]
        y_pred_cat = [cats[pred_val] for pred_val in y_pred]
        print('Predicted values: ', np.array(y_pred_cat))

    except pd.errors.ParserError as e:  # Hopefully works
        print('invalid test data')
        exit(1)

