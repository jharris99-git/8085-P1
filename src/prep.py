from ast import literal_eval

import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 5000)

protocols = ['udp', 'tcp', 'egp', 'unas', 'sun-nd', 'arp', 'idpr', 'gmtp', 'ipv6-route',
             'trunk-2', 'tlsp', 'bbn-rcc', 'sctp', 'ospf', 'dgp', 'ipcomp', 'prm', 'ptp',
             'crudp', 'hmp', 'kryptolan', 'wsn', 'ipv6', 'any', 'secure-vmtp', 'visa',
             'mobile', 'snp', 'iatp', 'wb-expak', 'crtp', 'il', 'ipnip', 'cpnx', 'fire',
             'mfe-nsp', 'pnni', 'leaf-1', 'xtp', 'xnet', 'vmtp', 'l2tp', 'ddp', 'narp', 'aris',
             '3pc', 'tcf', 'encap', 'ddx', 'swipe', 'idrp', 'iso-ip', 'ttp', 'sep',
             'compaq-peer', 'uti', 'pup', 'idpr-cmtp', 'iplt', 'gre', 'etherip', 'sprite-rpc',
             'sccopmce', 'qnx', 'ggp', 'ipv6-opts', 'ip', 'skip', 'br-sat-mon', 'a/n', 'mtp',
             'merit-inp', 'aes-sp3-d', 'ifmp', 'ipx-n-ip', 'cphb', 'leaf-2', 'vrrp', 'chaos',
             'st2', 'mhrp', 'sps', 'ib', 'trunk-1', 'mux', 'sat-mon', 'cftp', 'pim',
             'sat-expak', 'ippc', 'micp', 'rdp', 'pipe', 'larp', 'sdrp', 'ipv6-no', 'scps',
             'pri-enc', 'sm', 'nvp', 'nsfnet-igp', 'igp', 'isis', 'cbt', 'rvd', 'ipip',
             'ipv6-frag', 'pgm', 'pvp', 'emcon', 'bna', 'eigrp', 'rsvp', 'irtp', 'zero',
             'xns-idp', 'ipcv', 'icmp', 'srp', 'dcn', 'vines', 'smp', 'tp++', 'ax.25', 'netblt',
             'iso-tp4', 'stp', 'fc', 'argus', 'i-nlsp', 'wb-mon', 'esp', 'igmp']
proto_dtype = pd.CategoricalDtype(categories=protocols)

states = ['CON', 'FIN', 'INT', 'REQ', 'URH', 'RST', 'ECR', 'ECO', 'CLO', 'PAR', 'ACC', 'URN',
          'MAS']
state_dtype = pd.CategoricalDtype(categories=states)

services = ['dns', 'ftp', '-', 'http', 'smtp', 'ftp-data', 'ssh', 'pop3', 'dhcp', 'irc',
            'radius', 'ssl', 'snmp']
service_dtype = pd.CategoricalDtype(categories=services)


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
    data.sport = ['00' if x == "-" else x for x in data.sport.tolist()]
    data.sport = [literal_eval(x) if type(x) == str and x[0:2] == "0x" else int(x) for x in
                  data.sport.tolist()]

    data.dsport = ['00' if x == "-" else x for x in data.dsport.tolist()]
    data.dsport = [literal_eval(x) if type(x) == str and x[0:2] == "0x" else int(x) for x in
                   data.sport.tolist()]

    # Fix whitespace values in ct_ftp_cmd, to impute or prune later.
    data.ct_ftp_cmd = [None if x == " " else float(x) for x in data.ct_ftp_cmd.tolist()]

    # Fix attack_category whitespace and plural errors
    data.attack_cat = data.attack_cat.str.replace(" ", "")
    data.attack_cat = data.attack_cat.str.replace("Backdoors", "Backdoor")

    data = prune(data)
    data = encode(data)
    data = impute(data)
    return data


if __name__ == '__main__':
    # Define expected dtypes
    expected_dtypes = {'srcip': 'str',
                       'sport': 'str',
                       'dstip': 'str',
                       'dsport': 'str',
                       'proto': proto_dtype,
                       'state': state_dtype,
                       'service': service_dtype
                       }

    # Import base CSV data from file.
    base_data = pd.read_csv('../datasets/UNSW-NB15-BALANCED-TRAIN.csv', dtype=expected_dtypes,
                            low_memory=False)

    # Fix, Prune, encode, and impute.
    base_data = process_data(base_data)

    # print(base_data.proto.unique())
    # print(base_data.state.unique())
    # print(base_data.service.unique())

    print(base_data.dtypes)

    print(pd.read_csv('../datasets/prepped_label_data.csv', low_memory=False))

    print(pd.read_csv('../datasets/prepped_attack_cat_data.csv', low_memory=False))

    # Do this column dropping in train.py
    # final_prep(base_data)
