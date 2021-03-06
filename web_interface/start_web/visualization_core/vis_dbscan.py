import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go


def get_plot(model, data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    fig, ax = plt.subplots(figsize=(20, 18))

    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True
    labels = model.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.legend()
    return fig


def get_plot_2(model, data, model_columns):
    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True
    labels = model.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)

    fig = go.Figure()
    data = data[model_columns]

    for k in unique_labels:

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        X = xy.iloc[:, 0]
        Y = xy.iloc[:, 1]
        fig.add_traces(go.Scatter(x=X, y=Y, mode='markers'))

        xy = data[class_member_mask & ~core_samples_mask]
        X = xy.iloc[:, 0]
        Y = xy.iloc[:, 1]
        fig.add_traces(go.Scatter(x=X, y=Y, mode='markers'))

    return fig
