import os.path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS_TO_DROP = [" Form Name", "User Name", "id-hushed_internalpatientid"]


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    # global mean_values
    X = X.drop(columns=COLUMNS_TO_DROP)
    X = X[X.notnull()]
    # X = X.dropna(subset=[" Hospital"])
    # X[" Hospital"] = X[" Hospital"].astype(float)
    # X = X.reset_index(drop=True)
    duplicates = X.duplicated()
    X = X[~duplicates]
    # mean_values = X.mean(numeric_only=True)


    for col in df.columns.tolist():
        if 'Age' in col:
            X = X[X[col].isin([0, 130])]

    return X



def heat_map_correlation(X):
    '''
    This function creates a heatmap to understand the correlation between columns and understand which one to delete
    It prints the column that we need to delete (correlation >0.8)
    '''
    correlation = X.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(correlation, cmap='coolwarm', annot=True, ax=ax)

    # Find highly correlated features
    threshold = 0.8
    correlated_features = set()
    for i in range(len(correlation.columns)):
        for j in range(i):
            if abs(correlation.iloc[i, j]) > threshold:
                column_to_del = correlation.columns[i]
                correlated_features.add(column_to_del)

    # Print highly correlated features
    print("Highly correlated features to remove: ", correlated_features)
    fig.savefig('heatmap_correlation.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    np.random.seed(0)
    df = pd.read_csv("train.feats.csv",low_memory=False)
    # print(df.columns.tolist())
    df = preprocess_data(df)
    for col in df.columns.tolist():
        print(f"|{col}|")
        print(col, df[col].unique())
        print()


    # print(df["Hospital"])
    # print(df.columns.tolist())
    # df = preprocess_data(df)
    # heat_map_correlation(df)