import pickle
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

from prep import process_data


pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 5000)

SKLClassifier = Union[RandomForestClassifier, MLPClassifier, KNeighborsClassifier, SVC, LogisticRegression]

true_class = []
pred_class = []

ac_features = []
label_features = []


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
            est = RandomForestClassifier(n_estimators=2000, verbose=2, n_jobs=12)
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
            model = RandomForestClassifier(n_estimators=2000, verbose=2, n_jobs=12)
            kfold_means = train_score_model('Label', sel_label_data, model)

            # Print classification report of aggregated predictions.
            print(classification_report(y_true=true_class, y_pred=pred_class))

            # If the mean f1 score of kfold tests > 0.95, fit the model with more estimators and save the binary.
            if kfold_means > 0.95:
                y = sel_label_data['Label']
                x = sel_label_data.drop('Label', axis=1)

                # Fit final model.
                model_fin = RandomForestClassifier(n_estimators=4000, verbose=2, n_jobs=12)
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
            est = RandomForestClassifier(n_estimators=2000, verbose=2, n_jobs=10, class_weight='balanced_subsample')
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
            model = RandomForestClassifier(n_estimators=2000, verbose=2, n_jobs=12, class_weight='balanced_subsample')
            kfold_means = train_score_model('attack_cat', sel_ac_data, model)

            # Print classification report of aggregated predictions.
            print(classification_report(y_true=true_class, y_pred=pred_class))

            # If the mean f1 score of kfold tests > 0.95, fit the model with more estimators and save the binary.
            if kfold_means > 0.5:
                y = sel_ac_data['attack_cat']
                x = sel_ac_data.drop('attack_cat', axis=1)

                # Fit final model.
                model_fin = RandomForestClassifier(n_estimators=4000, verbose=2, n_jobs=10,
                                                   class_weight='balanced_subsample')
                model_fin.fit(x, y)

                # Pickle and save model as binary.
                save_pkl('attack_cat_RFC', model_fin)

            # Clear aggregated values.

            true_class = []
            pred_class = []


def feature_sel_test_K(data: pd.DataFrame, target: str):
    # Principal Component Analysis (PCA)
    # What it does: PCA reduces the dimensionality of the dataset by transforming the features into a smaller set of uncorrelated components. It identifies the most significant features by how much variance they explain.
    # How to use: You can use sklearn.decomposition.PCA to reduce the features and check how much variance each principal component explains.
    # checking shape
    global true_class
    global pred_class

    trainData = data.drop(['Label', 'attack_cat'], axis=1)
    data = pd.get_dummies(data, columns=['attack_cat'])
    # Input features
    X = trainData[trainData.columns[:-1]]
    # Mean
    X_mean = X.mean()
    # Standard deviation
    X_std = X.std()

    # Standardization
    Z = (X - X_mean) / X_std
    # covariance
    c = Z.cov()

    # Plot the covariance matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(50, 50))
    sns.heatmap(c, cmap="PiYG")
    # plt.show()

    eigenvalues, eigenvectors = np.linalg.eig(c)
    # Index the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    # Sort the eigenvalues in descending order
    eigenvalues = eigenvalues[idx]
    # sort the corresponding eigenvectors accordingly
    eigenvectors = eigenvectors[:, idx]
    explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = np.argmax(explained_var >= 0.50) + 1

    # PCA component or unit matrix
    u = eigenvectors[:, :n_components]
    pca_component = pd.DataFrame(u, index=trainData.columns[:-1])

    # plotting heatmap
    plt.figure(figsize=(20, 50))
    sns.heatmap(pca_component, cmap="PiYG")
    plt.title('PCA Component')
    # plt.show()

    # Matrix multiplication or dot Product
    Z_pca = Z @ pca_component

    # Importing PCA
    from sklearn.decomposition import PCA
    # Let's say, components = 2
    pca = PCA(n_components=n_components)
    pca.fit(Z)
    x_pca = pca.transform(Z)

    # Create the dataframe
    df_pca1 = pd.DataFrame(x_pca,
                           columns=['PC{}'.format(i + 1) for i in range(n_components)])
    # print(df_pca1)

    # giving a larger plot
    plt.figure(figsize=(8, 6))
    plt.plot(pca.explained_variance_ratio_)
    plt.show()
    plt.scatter(x_pca[:, 0], x_pca[:, 44],
                c=data['Label'],
                cmap='plasma')
    # labeling x and y axes
    plt.xlabel(str(0) + ' Principal Component')
    plt.ylabel(str(44) + ' Principal Component')
    # plt.show()

    sel_label_data = df_pca1
    sel_label_data['Label'] = data['Label']

    # Define model for kfold using selected features
    model = RandomForestClassifier(n_estimators=300, verbose=2, n_jobs=10)
    kfold_means = train_score_model('Label', sel_label_data, model)

    # Print classification report of aggregated predictions.
    print(classification_report(y_true=true_class, y_pred=pred_class))

    if kfold_means > 0.95:
        y = sel_label_data['Label']
        x = sel_label_data.drop('Label', axis=1)

        # Fit final model.
        model_fin = RandomForestClassifier(n_estimators=2000, verbose=2, n_jobs=10)
        model_fin.fit(x, y)
        # Pickle and save model as binary.
        save_pkl('Label_PCA', model_fin)

    true_class = []
    pred_class = []
    return df_pca1


def feature_sel_test_L(data: pd.DataFrame, target: str):
    # Drop the other target column (whichever is not selected)
    if target == 'attack_cat':
        ac_data = pd.DataFrame(data)
        ac_data = ac_data.drop(ac_data[ac_data.Label == 0].index, axis=0)
        ac_data = ac_data.drop(ac_data[ac_data['attack_cat'] == 0].index, axis=0)
        final_data = ac_data.drop('Label', axis=1)
    else:
        label_data = pd.DataFrame(data)
        final_data = label_data.drop('attack_cat', axis=1)

    # Separate the features (x) and the selected target (y)
    x = final_data  # Features (excluding the target labels)
    y = final_data[target]  # Target variable (either 'attack_cat' or 'Label')

    # Convert categorical features to numerical using LabelEncoder
    label_encoder = LabelEncoder()
    for column in x.select_dtypes(include=['object', 'category']).columns:
        x[column] = label_encoder.fit_transform(x[column])

    # Apply the chi-square test
    chi2_scores, p_values = chi2(x, y)

    # Create a DataFrame with the results
    chi2_results = pd.DataFrame({'Feature': x.columns, 'Chi2 Score': chi2_scores, 'p-value': p_values})

    # Sort by p-value for analysis
    chi2_results = chi2_results.sort_values(by='p-value')

    print(f"Chi-Square Test Results ({target}):")
    print(f"{chi2_results}\n")

    return chi2_results


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


def train_score_model(target_col: str, data: pd.DataFrame, model: SKLClassifier) -> (str, SKLClassifier):
    """

    :param target_col:
    :param data:
    :param model:
    :return:
    """
    y = data[target_col]
    x = data.drop(target_col, axis=1)

    # model = model.fit(x, y)

    kf = KFold(n_splits=5, shuffle=True, random_state=37)
    scores = cross_val_score(model, x, y, cv=kf, scoring=make_scorer(aggregating_f1_scorer))
    return np.mean(scores)


def save_pkl(name: str, model: SKLClassifier):
    """

    :param name:
    :param model:
    :return:
    """
    with open('../models/' + name + '.pkl', 'wb') as mdl_pkl:
        pickle.dump(model, mdl_pkl)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
if __name__ == '__main__':

    base_data = pd.read_csv('../datasets/UNSW-NB15-BALANCED-TRAIN.csv',
                            low_memory=False)


    base_data = process_data(base_data)

    NAME = ''

    match NAME:
        case 'J':
            feature_sel_test_J(base_data, 'Label')
        case 'K':
            feature_sel_test_K(base_data, 'Label')
        case 'L':
            # feature_sel_test_L(base_data, 'Label') # seems to be working fine
            feature_sel_test_L(base_data, 'attack_cat')
        case _:
            pass


