import gzip
import pickle
import time
from typing import Union

import numpy as np
import pandas as pd
from numpy.f2py.crackfortran import verbose
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, classification_report, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler

from prep import process_data


pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 5000)

SKLClassifier = Union[RandomForestClassifier, MLPClassifier, KNeighborsClassifier, SVC, LogisticRegression]

true_class = []
pred_class = []

ac_features = []
label_features = []


def experiment_j(data: pd.DataFrame, cat: str):
    # Feature Selection Method:     SelectFromModel
    # Chosen Classiifier:           RandomForestClassifier
    #
    # By default, uses the mean of the feature_importances_ values of a fit model to select features
    # based on the training of the model.
    #
    # Used here to select the features, then test those features across KFolds with F1-macro scoring.
    #
    # If the mean F1-macro score passes the given threshold, train a final RFC with more estimators and save it.

    # Used for aggregated classification report in KFold
    global true_class
    global pred_class

    # Split data for label classification
    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)

    # Split data for attack category classification
    ac_data = pd.DataFrame(data)
    ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
    ac_data = ac_data.drop('Label', axis=1)
    match cat:
        case 'Label':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            # ~~~~~~~~~~~~~~~~ Label ~~~~~~~~~~~~~~~~ #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            # Create RFC for fitting with no feature selection.
            mdl_1 = RandomForestClassifier(n_estimators=200, verbose=0, n_jobs=12)

            # Define features for training and scoring.
            sel_label_cols = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'dwin', 'dtcpb', 'smeansz', 'dmeansz', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'state_CLO', 'state_FIN', 'Label']

            # Create RFC for fitting with feature selection.
            mdl_2 = RandomForestClassifier(n_estimators=200, verbose=0, n_jobs=12)

            # Start timer
            start_time = time.time_ns()

            # Train and score general model, returning the average f1 score.
            kfold_means_1 = train_score_model('Label', label_data, mdl_1)
            print("No Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            # Clear aggregated values.
            true_class = []
            pred_class = []

            # Train and score fs model, returning the average f1 score.
            kfold_means_2 = train_score_model('Label', label_data[sel_label_cols], mdl_2)
            print("Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            # End time and calc diff
            end_time = time.time_ns()
            total_time = (end_time - start_time) / 1_000_000_000
            print(total_time, "s", sep='')

            # Compare average f1 scores, resolving which does better. Simpler model wins in case of a tie.
            result = "Feature Selection", "No Feature Selection" if kfold_means_2 - kfold_means_1 >= 0 else "No Feature Selection", "Feature Selection"
            print(result[0], "proved more reliable than", result[1])

            # Clear aggregated values.
            true_class = []
            pred_class = []

        case 'attack_cat':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            # ~~~~~~~~~~~ Attack Category ~~~~~~~~~~~ #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            # Factorize Attack Category
            factor = pd.factorize(ac_data['attack_cat'])
            ac_data.attack_cat = factor[0]
            definitions = factor[1]
            print(ac_data.attack_cat.head())
            print(definitions)

            # Create RFC for fitting with no feature selection.
            mdl_1 = RandomForestClassifier(n_estimators=180, verbose=0, n_jobs=10, class_weight='balanced_subsample')

            # Define features for training and scoring.
            sel_ac_cols = ['stcpb', 'dtcpb', 'Sload', 'Dload', 'dbytes', 'res_bdy_len', 'sbytes', 'Stime', 'Ltime', 'Djit', 'Sjit', 'dmeansz', 'sttl', 'Sintpkt', 'swin', 'dwin', 'Dpkts', 'Dintpkt', 'Spkts', 'dloss', 'ct_dst_src_ltm', 'ct_src_dport_ltm', 'ct_srv_dst', 'ct_srv_src', 'ct_dst_sport_ltm', 'dttl', 'smeansz', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_state_ttl', 'sloss', 'state_INT', 'proto_tcp', 'state_FIN', 'state_CON', 'service_dns', 'proto_udp', 'service_-', 'proto_unas', 'service_ftp-data', 'service_ssh', 'tcprtt', 'ct_flw_http_mthd', 'ct_ftp_cmd', 'ackdat', 'service_smtp', 'trans_depth', 'is_ftp_login', 'synack', 'attack_cat']

            # Create RFC for fitting with feature selection.
            mdl_2 = RandomForestClassifier(n_estimators=180, verbose=0, n_jobs=10, class_weight='balanced_subsample')

            # Start timer
            start_time = time.time_ns()

            # Train and score general model, returning the average f1 score.
            # kfold_means_1 = train_score_model('attack_cat', ac_data, mdl_1)
            # print("No Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            # Clear aggregated values.

            true_class = []
            pred_class = []

            # Train and score fs model, returning the average f1 score.
            kfold_means_2 = train_score_model('attack_cat', ac_data[sel_ac_cols], mdl_2)
            print("Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            # End time and calc diff
            end_time = time.time_ns()
            total_time = (end_time - start_time) / 1_000_000_000
            print(total_time, "s", sep='')

            # Compare average f1 scores, resolving which does better. Simpler model wins in case of a tie.
            result = "Feature Selection", "No Feature Selection" if kfold_means_2 - kfold_means_1 >= 0 else "No Feature Selection", "Feature Selection"
            print(result[0], "proved more reliable than", result[1])

            # Clear aggregated values.
            true_class = []
            pred_class = []

def experiment_K(data: pd.DataFrame, target: str):
    # Principal Component Analysis (PCA)
    # What it does: PCA reduces the dimensionality of the dataset by transforming the features into a smaller set of uncorrelated components. It identifies the most significant features by how much variance they explain.
    # How to use: You can use sklearn.decomposition.PCA to reduce the features and check how much variance each principal component explains.
    # checking shape
    global true_class
    global pred_class

    match target:
        case 'Label':
            label_data = pd.DataFrame(data)
            label_data = label_data.drop('attack_cat', axis=1)

            train_data_x = label_data.drop('Label', axis=1)
            train_data_y = label_data['Label']

            scaler = StandardScaler()
            train_data_x = scaler.fit_transform(train_data_x)

            # Importing PCA
            from sklearn.decomposition import PCA
            # Let's say, components = 15
            pca = PCA(n_components=50)
            pca.fit(train_data_x)
            pca_data = pca.transform(train_data_x)
            train_data = pd.DataFrame(pca_data)
            train_data[target] = train_data_y

            # Create RFC for use in a SelectFromModel feature selector and fit to determine column importance.
            mdl_1 = RandomForestClassifier(n_estimators=200, verbose=0, n_jobs=12)
            mdl_2 = RandomForestClassifier(n_estimators=200, verbose=0, n_jobs=12)

            kfold_means_1 = train_score_model('Label', label_data, mdl_1)
            print("No Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            true_class = []
            pred_class = []

            kfold_means_2 = train_score_model('Label', train_data, mdl_2)
            print("Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            result = "Feature Selection", "No Feature Selection" if kfold_means_2 - kfold_means_1 >= 0 else "No Feature Selection", "Feature Selection"
            print(result[0], "proved more reliable than", result[1])

            # Clear aggregated values.

            true_class = []
            pred_class = []
        case 'attack_cat':
            ack_data = pd.DataFrame(data)
            ack_data = ack_data.drop('Label', axis=1)
            factor = pd.factorize(ack_data['attack_cat'])
            ack_data.attack_cat = factor[0]

            train_data_x = ack_data.drop('attack_cat', axis=1)
            train_data_y = ack_data['attack_cat']

            scaler = StandardScaler()
            train_data_x = scaler.fit_transform(train_data_x)

            # Importing PCA
            from sklearn.decomposition import PCA
            # Let's say, components = 15
            pca = PCA(n_components=50)
            pca.fit(train_data_x)
            pca_data = pca.transform(train_data_x)
            train_data = pd.DataFrame(pca_data)
            train_data[target] = train_data_y

            # Create RFC for use in a SelectFromModel feature selector and fit to determine column importance.
            mdl_1 = RandomForestClassifier(n_estimators=200, verbose=0, n_jobs=12)
            mdl_2 = RandomForestClassifier(n_estimators=200, verbose=0, n_jobs=12)

            kfold_means_1 = train_score_model('attack_cat', ack_data, mdl_1)
            print("No Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            true_class = []
            pred_class = []

            kfold_means_2 = train_score_model('attack_cat', train_data, mdl_2)
            print("Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            result = "Feature Selection", "No Feature Selection" if kfold_means_2 - kfold_means_1 >= 0 else "No Feature Selection", "Feature Selection"
            print(result[0], "proved more reliable than", result[1])

            # Clear aggregated values.

            true_class = []
            pred_class = []



def experiment_l(data: pd.DataFrame, target: str):
    # Used for aggregated classification report in KFold
    global true_class
    global pred_class

    # Split data for label classification
    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)

    # Split data for attack category classification
    ac_data = pd.DataFrame(data)
    ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
    ac_data = ac_data.drop('Label', axis=1)

    match target:
        case 'Label':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            # ~~~~~~~~~~~~~~~~ Label ~~~~~~~~~~~~~~~~ #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            mdl_1 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
            mdl_1.fit(label_data.drop(target, axis=1), label_data[target])

            kfold_means_1 = train_score_model('Label', label_data, mdl_1)
            print("No Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            # Clear aggregated values.
            true_class = []
            pred_class = []

            sel_label_cols = ['stcpb', 'dtcpb', 'Sload', 'Dload', 'dbytes', 'res_bdy_len', 'sbytes', 'Stime', 'Ltime',
                              'Djit', 'Sjit', 'dmeansz', 'sttl', 'Sintpkt', 'swin', 'dwin', 'Dpkts', 'Dintpkt', 'Spkts',
                              'dloss', 'ct_dst_src_ltm', 'ct_src_dport_ltm', 'ct_srv_dst', 'ct_srv_src',
                              'ct_dst_sport_ltm', 'dttl', 'smeansz', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_state_ttl',
                              'sloss', 'state_INT', 'proto_tcp', 'state_FIN', 'state_CON', 'service_dns', 'proto_udp',
                              'service_-', 'proto_unas', 'service_ftp-data', 'service_ssh', 'tcprtt',
                              'ct_flw_http_mthd', 'ct_ftp_cmd', 'ackdat', 'service_smtp', 'trans_depth', 'is_ftp_login',
                              'synack', 'Label']

            mdl_2 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
            sel_label_data = label_data[sel_label_cols]
            # mdl_2.fit(sel_label_data.drop(target, axis=1), sel_label_data[target])

            # Normalize the data
            scaler = StandardScaler()
            scaled = scaler.fit_transform(sel_label_data)
            # Convert scaled arrays back to DataFrames
            sel_label_data_scaled_df = pd.DataFrame(scaled, columns=sel_label_data.columns)

            mdl_2.fit(sel_label_data_scaled_df.drop(target, axis=1), sel_label_data_scaled_df[target])
            kfold_means_2 = train_score_model('Label', sel_label_data_scaled_df, mdl_2)
            print("Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            result = "Feature Selection", "No Feature Selection" if kfold_means_2 - kfold_means_1 >= 0 else "No Feature Selection", "Feature Selection"
            print(result[0], "proved more reliable than", result[1])

            # Clear aggregated values.
            true_class = []
            pred_class = []

        case 'attack_cat':
            mdl_1 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=1, leaf_size=25, weights='distance')
            kfold_means_1 = train_score_model('attack_cat', ac_data, mdl_1)
            print("No Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class))

            # Clear aggregated values.
            true_class = []
            pred_class = []

            # Factorize Attack Category
            factor = pd.factorize(ac_data['attack_cat'])
            ac_data.attack_cat = factor[0]
            definitions = factor[1]

            mdl_2 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=1, leaf_size=25, weights='distance')
            sel_ac_cols = ['dtcpb', 'stcpb', 'Sload', 'Dload', 'sbytes', 'Sjit', 'dbytes', 'res_bdy_len', 'Djit',
                           'Sintpkt', 'Dintpkt', 'dmeansz', 'swin', 'dwin', 'dttl', 'smeansz', 'Spkts', 'Dpkts',
                           'Stime', 'Ltime', 'sloss', 'ct_dst_src_ltm', 'ct_srv_dst', 'ct_srv_src', 'dloss',
                           'ct_src_dport_ltm', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_dst_sport_ltm', 'sttl', 'dur',
                           'service_-', 'proto_tcp', 'state_FIN', 'service_dns', 'attack_cat']

            sel_ac_data = ac_data[sel_ac_cols]
            # Normalize the data
            scaler = StandardScaler()
            scaled = scaler.fit_transform(sel_ac_data.drop(columns=target))
            # Convert scaled arrays back to DataFrames
            sel_ac_data_scaled_df = pd.DataFrame(scaled, columns=sel_ac_data.columns.drop(target))
            sel_ac_data_scaled_df['attack_cat'] = sel_ac_data['attack_cat'].values
            # Ensure the target variable is integer type
            sel_ac_data_scaled_df[target] = sel_ac_data_scaled_df[target].astype(int)

            mdl_2.fit(sel_ac_data_scaled_df.drop(target, axis=1), sel_ac_data_scaled_df[target])
            kfold_means_2 = train_score_model('attack_cat', sel_ac_data_scaled_df, mdl_2)
            print("Feature Selection:\n", classification_report(y_true=true_class, y_pred=pred_class,
                                                                target_names=definitions))

            result = "Feature Selection", "No Feature Selection" if kfold_means_2 - kfold_means_1 >= 0 else "No Feature Selection", "Feature Selection"
            print(result[0], "proved more reliable than", result[1])

            # Clear aggregated values.
            true_class = []
            pred_class = []


def feature_sel_test_J(data: pd.DataFrame, cat: str):
    # Feature Selection Method:     SelectFromModel
    # Chosen Classiifier:           RandomForestClassifier
    #
    # By default, uses the mean of the feature_importances_ values of a fit model to select features
    # based on the training of the model.
    #
    # Used here to select the features, then test those features across KFolds with F1-macro scoring.
    #
    # If the mean F1-macro score passes the given threshold, train a final RFC with more estimators and save it.

    # Used for aggregated classification report in KFold
    global true_class
    global pred_class

    # Split data for label classification
    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)

    # Split data for attack category classification
    ac_data = pd.DataFrame(data)
    ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
    ac_data = ac_data.drop('Label', axis=1)

    match cat:
        case 'Label':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            # ~~~~~~~~~~~~~~~~ Label ~~~~~~~~~~~~~~~~ #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            # ~~~~~~~ Label Feature Selection ~~~~~~~ #

            label_data_y = label_data['Label']
            label_data_X = label_data.drop('Label', axis=1)

            # Scale data, not always necessary.
            # scaler = StandardScaler().fit(label_data_X)
            # label_data_X = scaler.transform(label_data_X)

            # Create RFC for use in a SelectFromModel feature selector and fit to determine column importance.
            est = RandomForestClassifier(n_estimators=200, verbose=2, n_jobs=12)
            sel = SelectFromModel(est)

            sel = sel.fit(label_data_X, label_data_y)

            # Get selected labels according to SelectFromModel and add them to the columns for use in the final model
            sel_feat_ind_Label = np.where(sel.get_support())[0]
            sel_feat_Label = label_data.columns[sel_feat_ind_Label]

            print(sel_feat_Label)

            sel_label_data_cols = sel_feat_Label.to_list() + ['Label']
            sel_label_data = label_data[sel_label_data_cols]

            # ~~~~~~~ Label Model Training ~~~~~~~ #

            # Define model for kfold using selected features
            model = RandomForestClassifier(n_estimators=200, verbose=2, n_jobs=12)
            kfold_means = train_score_model('Label', sel_label_data, model)

            # Print classification report of aggregated predictions.
            print(classification_report(y_true=true_class, y_pred=pred_class))

            # If the mean f1 score of kfold tests > 0.95, fit the model with more estimators and save the binary.
            if kfold_means > 0.95:
                y = sel_label_data['Label']
                x = sel_label_data.drop('Label', axis=1)

                # Fit final model.
                model_fin = RandomForestClassifier(n_estimators=200, verbose=2, n_jobs=12)
                model_fin.fit(x, y)
                # Pickle and save model as binary.
                save_pkl('Label_RFC', model_fin)

            # Clear aggregated values.

            true_class = []
            pred_class = []

        case 'attack_cat':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            # ~~~~~~~~~~~ Attack Category ~~~~~~~~~~~ #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            # ~~ Attack Category Feature Selection ~~ #

            # Factorize Attack Category
            factor = pd.factorize(ac_data['attack_cat'])
            ac_data.attack_cat = factor[0]
            definitions = factor[1]
            print(ac_data.attack_cat.head())
            print(definitions)

            ac_data_y = ac_data['attack_cat']
            ac_data_X = ac_data.drop('attack_cat', axis=1)

            # Create RFC for use in a SelectFromModel feature selector and fit to determine column importance.
            est = RandomForestClassifier(n_estimators=180, verbose=2, n_jobs=10, class_weight='balanced_subsample')
            sel = SelectFromModel(est)  # , threshold=-np.inf, max_features=60

            sel = sel.fit(ac_data_X, ac_data_y)

            # Get selected Attack Categories according to SelectFromModel /
            # and add them to the columnsfor use in the final model
            sel_feat_ind_ac = np.where(sel.get_support())[0]
            sel_feat_ac = ac_data.columns[sel_feat_ind_ac]

            print(sel_feat_ac)

            sel_ac_data_cols = sel_feat_ac.to_list() + ['attack_cat']
            sel_ac_data = ac_data[sel_ac_data_cols]

            # ~~ Attack  Category  Model  Training ~~ #

            # Define model for kfold using selected features
            model = RandomForestClassifier(n_estimators=180, verbose=2, n_jobs=12, class_weight='balanced_subsample')
            kfold_means = train_score_model('attack_cat', sel_ac_data, model)

            # Print classification report of aggregated predictions.
            print(classification_report(y_true=true_class, y_pred=pred_class))

            # If the mean f1 score of kfold tests > 0.95, fit the model with more estimators and save the binary.
            if kfold_means > 0.5:
                y = sel_ac_data['attack_cat']
                x = sel_ac_data.drop('attack_cat', axis=1)

                # Fit final model.
                model_fin = RandomForestClassifier(n_estimators=180, verbose=2, n_jobs=10,
                                                   class_weight='balanced_subsample')
                model_fin.fit(x, y)

                # Pickle and save model as binary.
                save_pkl('attack_cat_RFC', model_fin)

            # Clear aggregated values.

            true_class = []
            pred_class = []

def feature_sel_test_K(data: pd.DataFrame, target: str):
    train_data_x = None
    train_data_y = None
    match target:
        case 'Label':
            label_data = pd.DataFrame(data)
            label_data = label_data.drop('attack_cat', axis=1)

            train_data_x = label_data.drop('Label', axis=1)
            train_data_y = label_data['Label']
        case 'attack_cat':
            ack_data = pd.DataFrame(data)
            ack_data = ack_data.drop('Label', axis=1)
            factor = pd.factorize(ack_data['attack_cat'])
            ack_data.attack_cat = factor[0]

            train_data_x = ack_data.drop('attack_cat', axis=1)
            train_data_y = ack_data['attack_cat']

    # Uncomment to generate models
    global true_class
    global pred_class

    scaler = StandardScaler()
    train_data_x = scaler.fit_transform(train_data_x)
    train_data = pd.DataFrame(train_data_x)
    train_data[target] = train_data_y

    # model = MLPClassifier(max_iter=300, verbose=1)
    # GRID = [
    #     {
    #      'solver': ['adam'],
    #      'hidden_layer_sizes': [(500, 400, 300, 200, 100), (400, 400, 400, 400, 400),
    #                                        (300, 300, 300, 300, 300), (200, 200, 200, 200, 200)],
    #      'activation': ['logistic', 'tanh', 'relu'],
    #      'alpha': [0.0001, 0.001],
    #      'early_stopping': [True]
    #      }
    # ]
    # random_search = RandomizedSearchCV(model, GRID,
    #                            scoring=make_scorer(accuracy_score, average='macro'),
    #                            n_jobs=-1, cv=3, refit=True, verbose=1)
    # random_search.fit(train_data, train_data_y)
    # print(random_search.best_params_)

    model = MLPClassifier(solver='adam', hidden_layer_sizes=(400, 400, 400, 400, 400), alpha=0.001, activation='relu',early_stopping=True, max_iter=300, verbose=1)
    # # Define model for kfold using selected features
    kfold_means = train_score_model(target, train_data, model)

    # # Print classification report of aggregated predictions.
    print(classification_report(y_true=true_class, y_pred=pred_class))

    # # If the mean f1 score of kfold tests > 0.95, fit the model with more estimators and save the binary.
    if kfold_means > 0.95:
        y = train_data[target]
        x = train_data.drop(target, axis=1)

        # Fit final model.
        model_fin = MLPClassifier(solver='adam', hidden_layer_sizes=(400, 400, 400, 400, 400), alpha=0.0001, activation='relu',early_stopping=True, max_iter=400, verbose=1)
        model_fin.fit(x, y)
        # Pickle and save model as binary.
        save_pkl(target + '_MLP', model_fin)

    true_class = []
    pred_class = []
    return train_data_x


def feature_sel_test_L(data: pd.DataFrame, target: str):
    # Used for aggregated classification report in KFold
    global true_class
    global pred_class

    # Select Data
    ac_data = pd.DataFrame(data)
    ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
    ac_data = ac_data.drop(ac_data[ac_data['attack_cat'] == 0].index, axis=0)
    ac_data = ac_data.drop('Label', axis=1)

    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)

    match target:
        case 'Label':
            # Separate the features (x) and the selected target (y)
            x = label_data.drop(target, axis=1)  # Features (excluding the target labels)
            y = label_data[target]  # Target variable (either 'attack_cat' or 'Label')

            # Apply the chi-square test
            chi2_scores, p_values = chi2(x, y)
            # Create a DataFrame with the results
            chi2_results = pd.DataFrame({'Feature': x.columns, 'Chi2 Score': chi2_scores, 'p-value': p_values})
            # Filter results with p-value = 0 & highest chi2 Score
            sel_chi2_results = chi2_results[chi2_results['p-value'] == 0]
            sel_chi2_results = sel_chi2_results.sort_values(by='Chi2 Score', ascending=False)

            # Output results
            print(f'Chi-Square Test Results ({target}):')
            print(f'{chi2_results}')
            print(f'============================================================================')
            print(f"Top {len(sel_chi2_results)} features selected with 0 p-value & sorted by highest chi2 scores:")
            print(sel_chi2_results)

            # Get selected feature names
            selected_features = sel_chi2_results['Feature'].values
            sel_label_data_cols = selected_features.tolist() + ['Label']
            sel_label_data = label_data[sel_label_data_cols]

            # ~~~~~~~ Label Model Training ~~~~~~~ #
            # Normalize the data
            scaler = StandardScaler()
            scaled = scaler.fit_transform(sel_label_data)
            # Convert scaled arrays back to DataFrames
            sel_label_data_scaled_df = pd.DataFrame(scaled, columns=sel_label_data.columns)

            # Define model for kfold using selected features
            model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
            model.fit(sel_label_data_scaled_df.drop(target, axis=1), sel_label_data_scaled_df[target])
            kfold_means = train_score_model(target, sel_label_data_scaled_df, model)

            # Print classification report of aggregated predictions.
            print(classification_report(y_true=true_class, y_pred=pred_class))

            # If the mean f1 score of kfold tests > 0.95, fit the model with more estimators and save the binary.
            # if kfold_means > 0.95:
            #     final_y = sel_label_data_scaled_df[target]
            #     final_x = sel_label_data_scaled_df.drop(target, axis=1)
            #
            #     # Fit final model.
            #     model_fin = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
            #     model_fin.fit(final_x, final_y)
            #     # Pickle and save model as binary.
            #     save_pkl('Label_CHI', model_fin)

            # Clear aggregated values.
            true_class = []
            pred_class = []

        case 'attack_cat':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            # ~~~~~~~~~~~ Attack Category ~~~~~~~~~~~ #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            # ~~ Attack Category Feature Selection ~~ #

            # Factorize Attack Category
            factor = pd.factorize(ac_data['attack_cat'])
            ac_data.attack_cat = factor[0].astype(int)
            definitions = factor[1]
            print(ac_data.attack_cat.head())
            print(definitions)

            # Separate features and target
            y = ac_data[target]
            x = ac_data.drop(target, axis=1)

            # Convert categorical features to numerical using LabelEncoder
            label_encoder = LabelEncoder()
            for column in x.select_dtypes(include=['object', 'category']).columns:
                x[column] = label_encoder.fit_transform(x[column])

            # Remove constant columns (those with zero variance)
            x = x.loc[:, (x != x.iloc[0]).any()]

            # Apply the chi-square test
            chi2_scores, p_values = chi2(x, y)
            # Create a DataFrame with the results
            chi2_results = pd.DataFrame({'Feature': x.columns, 'Chi2 Score': chi2_scores, 'p-value': p_values})
            # Filter results with p-value = 0 & highest chi2 Score
            sel_chi2_results = chi2_results[chi2_results['p-value'] == 0]
            sel_chi2_results = sel_chi2_results.sort_values(by='Chi2 Score', ascending=False)
            sel_chi2_results = sel_chi2_results.head(35) #seems like a good cutoff

            # Output results
            print(f'Chi-Square Test Results ({target}):')
            print(f'{chi2_results}')
            print(f'============================================================================')
            print(f"Top {len(sel_chi2_results)} features selected with 0 p-value:")
            print(sel_chi2_results)

            # Get selected feature names
            selected_features = sel_chi2_results['Feature'].values
            sel_ac_data_cols = selected_features.tolist() + [target]
            sel_ac_data = ac_data[sel_ac_data_cols]

            # ~~ Attack Category Model Training ~~ #

            # Normalize the data
            scaler = StandardScaler()
            scaled = scaler.fit_transform(sel_ac_data.drop(columns=target))

            # Convert scaled arrays back to DataFrames
            sel_ac_data_scaled_df = pd.DataFrame(scaled, columns=sel_ac_data.columns.drop(target))
            sel_ac_data_scaled_df['attack_cat'] = sel_ac_data['attack_cat'].values

            # Ensure the target variable is integer type
            sel_ac_data_scaled_df[target] = sel_ac_data_scaled_df[target].astype(int)

            model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=1, leaf_size=25, weights='distance')
            model.fit(sel_ac_data_scaled_df.drop(target, axis=1), sel_ac_data_scaled_df[target])
            kfold_means = train_score_model(target, sel_ac_data_scaled_df, model, 30)
            print(classification_report(y_true=true_class, y_pred=pred_class, target_names=definitions))

            # if kfold_means > 0.45:
            #     final_y = sel_ac_data_scaled_df[target]
            #     final_x = sel_ac_data_scaled_df.drop(target, axis=1)
            #
            #     model_fin = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=1, leaf_size=25, weights='distance')
            #     model_fin.fit(final_x, final_y)
            #     save_pkl('attack_cat_CHI', model_fin)

            true_class = []
            pred_class = []


def feature_select(data: pd.DataFrame, features: list):
    """

    :param data:
    :param features:
    :return:
    """
    # remove unnecessary features
    test_data = data[features]
    return test_data


def initial_prep_attack_cat(data: pd.DataFrame):
    """

    :param data:
    :return:
    """
    ac_data = pd.DataFrame(data)
    ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
    ac_data = ac_data.drop('Label', axis=1)
    return ac_data


def initial_prep_label(data: pd.DataFrame):
    """

    :param data:
    :return:
    """
    label_data = pd.DataFrame(data)
    label_data = label_data.drop('attack_cat', axis=1)
    return label_data


def aggregating_f1_scorer(y_true, y_pred):
    true_class.extend(y_true)
    pred_class.extend(y_pred)
    return f1_score(y_true, y_pred, average='macro')


def train_score_model(target_col: str, data: pd.DataFrame, model: SKLClassifier, folds: int = 5):
    """

    :param target_col:
    :param data:
    :param model:
    :param folds:
    :return:
    """
    y = data[target_col]
    x = data.drop(target_col, axis=1)

    # model = model.fit(x, y)

    kf = KFold(n_splits=folds, shuffle=True, random_state=37)
    scores = cross_val_score(model, x, y, cv=kf, scoring=make_scorer(aggregating_f1_scorer))
    return np.mean(scores)


def save_pkl(name: str, model: SKLClassifier):
    """

    :param name:
    :param model:
    :return:
    """
    with gzip.open('../models/' + name + '.pkl', 'wb') as mdl_pkl:
        pickle.dump(model, mdl_pkl, protocol=pickle.HIGHEST_PROTOCOL)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
if __name__ == '__main__':

    base_data = pd.read_csv('../datasets/UNSW-NB15-BALANCED-TRAIN.csv',
                            low_memory=False)


    base_data = process_data(base_data)

    NAME = 'J'

    match NAME:
        case 'J':
            # feature_sel_test_J(base_data, 'attack_cat')
            experiment_j(base_data, 'attack_cat')
        case 'K':
            # feature_sel_test_K(base_data, 'Label')
            # feature_sel_test_K(base_data, 'attack_cat')
            # experiment_K(base_data, 'Label')
            experiment_K(base_data, 'attack_cat')
        case 'L':
            # feature_sel_test_L(base_data, 'Label')
            # feature_sel_test_L(base_data, 'attack_cat')
            # experiment_l(base_data, 'Label')
            experiment_l(base_data, 'attack_cat')
        case _:
            pass


