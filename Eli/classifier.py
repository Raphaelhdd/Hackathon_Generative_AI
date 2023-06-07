import ast
import pandas as pd
import sklearn.model_selection
import numpy as np

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor



def load_data(data):
    working_data = traduct_data(data[["אבחנה-Age", "אבחנה-Margin Type", "אבחנה-Side", "אבחנה-er", 'אבחנה-Her2',
                         'אבחנה-M -metastases mark (TNM)','אבחנה-N -lymph nodes mark (TNM)', 'אבחנה-T -Tumor mark (TNM)']])
    working_data = pd.get_dummies(working_data)
    starting_data = data[["אבחנה-Age", "אבחנה-Margin Type", "אבחנה-Side"]]
    starting_data = pd.get_dummies(starting_data)
    for col in working_data.columns:
        if working_data[col].isnull().values.any():
            complet_column = fill_data(starting_data, working_data[col])
            working_data[col] = complet_column
            starting_data[col] = complet_column
            starting_data = pd.get_dummies(starting_data)
    return working_data



def traduct_data(data_pd):

    def get_her2(data):
        labels = {
            "POS": list(),
            "NEG": list(),
            np.NaN: list(),
        }

        for value in data["אבחנה-Her2"].unique():
            try:
                CONTINUE = False

                for positive in [
                    '=', 'pos', 'po', 'yes', '3', '100',
                    "amplified", "amp", "חיובי", '+', 'o'
                ]:
                    if (positive in value.lower() and
                        ("neg" not in value.lower() and
                         "not" not in value.lower() and
                         "non" not in value.lower() and
                         "no" not in value.lower() and
                         "שלילי" not in value.lower() and
                         "1/30%" not in value.lower())):
                        labels["POS"].append(value)

                        CONTINUE = True
                        break

                if CONTINUE:
                    continue

                for negative in ['_', 'no', 'neg', 'non', '1', '2', '+@', 'nef',
                                 ',eg', 'meg', 'nrg', 'nec', 'nag', 'nfg',
                                 ')', "fish", '0', '-', 'not',
                                 "בינוני", "שלילי"]:
                    if negative in value.lower():
                        labels["NEG"].append(value)

                        CONTINUE = True
                        break

                if CONTINUE:
                    continue

                else:
                    labels[np.NaN].append(value)

            except:
                labels[np.NaN].append(value)

        return labels

    def get_pr(data, feature):
        dct = {100: [], 90: [], 80: [], 70: [], 60: [], 50: [],
               40: [], 30: [], 20: [], 10: [], 0: [], np.NaN: [None]}

        # print(data.columns)
        for value in data[feature].unique():
            added = False
            try:
                for i in range(100, -1, -1):
                    if str(i) in value:
                        lst = (i // 10) * 10
                        dct[int(lst)].append(value)
                        added = True
                        break
                if added:
                    continue

                for x in ["POSITIVE", "pos", "positive", "חיובי", "POS", "Positive", "+", "high"]:
                    if x.lower() in value.lower():
                        dct[100].append(value)
                        added = True
                        break
                    if value.lower() in x.lower():
                        dct[100].append(value)
                        added = True
                        break
                if added:
                    continue

                for x in ["NEGATIVE", "neg", "negative", "שלילי", "NEG", "Negative", "-", "_", "low"]:
                    if value.lower() in x.lower():

                        dct[0].append(value)
                        added = True
                        break
                    elif x.lower() in value.lower():
                        dct[0].append(value)
                        added = True
                        break
                if added:
                    continue

                dct[np.NaN].append(value)

            except:
                dct[np.NaN].append(value)

        return dct

    labels_her2 = get_her2(data_pd)
    labels_er = get_pr(data_pd, 'אבחנה-er')
    # labels_pr = get_pr(data_pd, 'אבחנה-pr')
    #labels_KI67 = get_KI67(data_pd)

    for key in labels_her2:
        new_values = labels_her2[key]

        data_pd['אבחנה-Her2'] = data_pd['אבחנה-Her2'].replace(new_values, key)

    for key in labels_er:
        new_values = labels_er[key]

        data_pd['אבחנה-er'] = data_pd['אבחנה-er'].replace(new_values, key)

    # for key in labels_pr:
    #     new_values = labels_er[key]
    #
    #     data_pd['אבחנה-pr'] = data_pd['אבחנה-pr'].replace(new_values, key)

    # for key in labels_KI67:
    #     new_values = labels_KI67[key]
    #
    #     data_pd['אבחנה-KI67 protein'] = data_pd['אבחנה-KI67 protein'].replace(new_values, key)

    data_pd['אבחנה-M -metastases mark (TNM)'] = data_pd['אבחנה-M -metastases mark (TNM)'].replace(["Not yet Established", None], np.NaN)

    data_pd['אבחנה-N -lymph nodes mark (TNM)'] = data_pd['אבחנה-N -lymph nodes mark (TNM)'].replace(["Not yet Established", None, "#NAME?"], np.NaN)

    # data_pd['אבחנה-Stage'] = data_pd['אבחנה-Stage'].replace(["Not yet Established", None, "#NAME?"], np.NaN)

    # data_pd['אבחנה-Surgery date1'] = data_pd['אבחנה-Surgery date1'].replace(["Unknown", "Not yet Established", None, "#NAME?"], np.NaN)

    data_pd['אבחנה-T -Tumor mark (TNM)'] = data_pd['אבחנה-T -Tumor mark (TNM)'].replace(["Unknown", "Not yet Established", None, "#NAME?"], np.NaN)
    return data_pd

def fill_data(X, y):
    X_train = X[y.notna()]
    y_train = y[y.notna()]

    X_test = X[y.isna()]
    knn = KNeighborsClassifier().fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y[y.isna()] = y_pred
    return y


if __name__ == '__main__':
    features_train_path = 'Mission 2 - Breast Cancer/train.feats.csv'
    features_train_data = pd.read_csv(features_train_path)
    X_train = load_data(features_train_data)

    labels_train_path = "Mission 2 - Breast Cancer/train.labels.0.csv"
    y_train = pd.read_csv(labels_train_path)
    y_train = y_train["אבחנה-Location of distal metastases"].apply(ast.literal_eval).to_numpy()
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)

    features_test_path = 'Mission 2 - Breast Cancer/test.feats.csv'
    features_test_data = pd.read_csv(features_test_path)
    X_test = load_data(features_test_data)

    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    for col in X_test.columns:
        if col not in X_train.columns:
            X_train[col] = 0

    clf = MultiOutputClassifier(DecisionTreeClassifier()).fit(X_train, y_train_bin)

    y_pred_bin = clf.predict(X_test)
    y_pred = pd.DataFrame(mlb.inverse_transform(y_pred_bin))
    y_pred = y_pred.values.tolist()
    df1 = pd.DataFrame({'final': y_pred})
    df1['final'] = [[x for x in inner_list if x is not None] for inner_list in df1['final']]
    pd.DataFrame(df1).to_csv("Mission 2 - Breast Cancer/y_pred_0.csv", header=False, index=False)

    # Part 2:
    labels_train_path = "Mission 2 - Breast Cancer/train.labels.1.csv"
    y_train = pd.read_csv(labels_train_path)
    y_train = y_train["אבחנה-Tumor size"].to_numpy(int)

    lr = GradientBoostingRegressor().fit(X_train, y_train)
    y_pred = np.maximum(0, np.round(lr.predict(X_test), 1))
    pd.DataFrame({"אבחנה-Tumor size": y_pred}).to_csv("Mission 2 - Breast Cancer/y_pred_1.csv", index=False)


