import os.path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS_TO_DROP = []


def heat_map_correlation():
    '''
    This function creates a heatmap to understand the correlation between columns and understand which one to delete
    It prints the column that we need to delete (correlation >0.8)
    '''
    correlation = df.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(correlation, cmap='coolwarm', annot=True, ax=ax)

    # Find highly correlated features
    threshold = 0.6
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
    df = pd.read_csv("train.feats.csv")
    df = df.drop(columns=COLUMNS_TO_DROP)

    heat_map_correlation()