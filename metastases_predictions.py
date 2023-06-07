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
    #1. Drop useless columns
    X = X.drop(columns=COLUMNS_TO_DROP)
    #2. Check in each column how many null there is according to the true data and drop if many null
    null_counts = {}
    not_null_counts = {}
    for column in df.columns:
        null_counts[column] = df[column].isnull().sum()
        not_null_counts[column] = df[column].notnull().sum()

    for column in df.columns:
        print(f"Column '{column}': {null_counts[column]} null values, {not_null_counts[column]} non-null values")

    # X = X[X.notnull()]
    # X = X.dropna(subset=[" Hospital"])
    # X[" Hospital"] = X[" Hospital"].astype(float)
    # X = X.reset_index(drop=True)
    # duplicates = X.duplicated()
    # X = X[~duplicates]
    # mean_values = X.mean(numeric_only=True)


    for col in df.columns.tolist():
        if 'Age' in col:
            X = X[X[col].isin([0, 130])]

    return X



def heat_map_correlation(correlation):
    '''
    This function creates a heatmap to understand the correlation between columns and understand which one to delete
    It prints the column that we need to delete (correlation >0.8)
    '''
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


def replace_non_numeric_null(df, non_numeric_columns):
    dict_col = dict()
    for col in non_numeric_columns:
        dict_val = dict()
        for val in df[col].unique():
            count_val = df[col].value_counts()
            dict_val[val] = count_val
        dict_col[col] = dict_val
    print(dict_col)

if __name__ == "__main__":
    np.random.seed(0)
    df = pd.read_csv("train.feats.csv",low_memory=False)
    # df = preprocess_data(df)
    # 1. Drop useless columns
    df = df.drop(columns=COLUMNS_TO_DROP)
    # 2. Check in each column how many null there is according to the true data and drop if many null
    null_counts = {}
    not_null_counts = {}
    for column in df.columns:
        null_counts[column] = df[column].isnull().sum()
        not_null_counts[column] = df[column].notnull().sum()

    # Drop column that the number of null is too big
    columns_null_to_drop = columns_null(df)
    df = df.drop(columns=columns_null_to_drop)

    # Change numeric values column by the mean of the feature
    numeric_columns = df.select_dtypes(include=['number']).columns
    df = replace_null_with_mean(df, numeric_columns)

    # Change non-numeric values column by the mean of the feature if very
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    df = replace_non_numeric_null(df, non_numeric_columns)

