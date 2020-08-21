import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vispy import plot as vp

import plotly.express as px
import plotly.graph_objects as go

import numpy as np

import io
import urllib, base64

import visualization_core.vis_dbscan as v_dbscan
import visualization_core.vis_kmeans as v_kmeans
import visualization_core.vis_aggcluster as v_aggcluster
import visualization_core.vis_RFregressor as v_RFregressor
import visualization_core.vis_GPregressor as v_GPregressor
import visualization_core.vis_LinRegression as v_Lregressor


# Входная точка для визуализации данных
def data_plot(data, params):
    type = params["type"]
    if type == "plot":
        return get_plot(data, params["draw"], params["plot_size"])
    if type == "hist":
        return get_hist(data, params["bins"], params["plot_size"])
    if type == "heatmap":
        return get_heatmap(data, params["plot_size"])
    if type == "fill_between":
        return get_fill_between(data, params["plot_size"])
    return None


# Входная точка для визуализации данных
def data_plotly(data, params):
    type = params["type"]
    if type == "plot":
        return get_plot_2(data, params["draw"])
    if type == "hist":
        return get_hist_2(data, params["bins"])
    if type == "heatmap":
        return get_heatmap_2(data)
    if type == "boxplot":
        return get_boxplot_2(data)
    if type == "3D":
        return get_chart_3d(data)
    if type == "3Dsurface":
        return get_chart_3dsurface(data)
    if type == "fill_between":
        return get_fill_between(data, params["plot_size"])
    return None


# Визуализация простого графика
def get_plot(data, draw, plot_size):
    width, height = get_plot_size(plot_size)
    fig, ax = plt.subplots(figsize=(width, height))
    # plt.grid(True)
    columns_count = len(data.columns)

    if columns_count < 2:
        return fig

    if columns_count == 2:
        x_label = data.columns.values[0]
        y_label = data.columns.values[1]
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.plot(data[x_label], data[y_label], draw)

    if columns_count > 2:
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        x_label = data.columns.values[0]
        ax.set_xlabel(x_label)
        mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:grey', 'tab:cyan']
        columns = data.columns[1:columns_count]
        date_column = data.columns[0:1][0]
        for i, column in enumerate(columns):
            label = ''.join(filter(whitelist.__contains__, column))
            plt.plot(data[date_column].values, data[column].values, draw, lw=1.5, color=mycolors[i], label=label)
        plt.legend(loc='upper left')

    return fig


# Визуализация простого графика
def get_plot_2(data, draw):
    fig = go.Figure()
    if len(data.columns) < 2:
        return fig
    date_column = data.columns[0]
    for column in data.columns[1:]:
        fig.add_traces(go.Scatter(x=data[date_column].values, y=data[column].values, name=column, mode=draw))

    return fig


# Построение гистограммы
def get_hist(data, bins, plot_size):
    width, height = get_plot_size(plot_size)
    fig, ax = plt.subplots(figsize=(width, height))
    plt.grid(True)
    x_label = data.columns.values[0]
    ax.set_xlabel(x_label)
    plt.hist(data[x_label], bins=bins, facecolor='blue', alpha=0.75)
    return fig


# Построение гистограммы
def get_hist_2(data, bins):
    x_label = data.columns.values[0]
    fig = px.histogram(data, x=x_label, nbins=bins)
    return fig


# Построение диаграммы корреляции
def get_heatmap(data, plot_size):
    width, height = get_plot_size(plot_size)
    fig, ax = plt.subplots(figsize=(width, height))
    sns_plot = sns.heatmap(data.corr(), xticklabels=data.corr().columns,
                           yticklabels=data.corr().columns, cmap='RdYlGn', center=0, annot=True, square=True,
                           linewidths=.5, ax=ax, robust=True)
    plt.title('Корреляция', fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    fig = sns_plot.get_figure()
    return fig


# Построение диаграммы корреляции
def get_heatmap_2(data):
    fig = px.imshow(data.values)
    return fig


# Построение диаграммы с усами
def get_boxplot_2(data):
    fig = px.box(data, y=data.columns[0], points="all")
    return fig


def get_chart_3d(data):
    fig = go.Figure(data=[go.Mesh3d(x=data.iloc[:,0],
                                    y=data.iloc[:,1],
                                    z=data.iloc[:,2],
                                    opacity=0.5,
                                    color='rgba(244,22,100,0.6)'
                                    )])
    fig.update_layout(scene = dict(
                    xaxis_title=data.columns[0],
                    yaxis_title=data.columns[1],
                    zaxis_title=data.columns[2]))
    return fig


def get_chart_3dsurface(data):
    fig = go.Figure(data=[go.Surface(z=data.values)])
    fig.update_layout(title=data.columns[0])
    return fig


# Построение диаграммы доверительного интервала
def get_fill_between(data, plot_size):
    width, height = get_plot_size(plot_size)
    fig, ax = plt.subplots(figsize=(width, height))
    plt.grid(True)
    columns_count = len(data.columns)
    try:
        if columns_count == 2:
            x_label = data.columns.values[0]
            y_label = data.columns.values[1]
            x = data[x_label].astype('int32')
            y = data[y_label]
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            a, b = np.polyfit(x, y, 1)
            y_est = a * x + b
            y_err = x.std() * np.sqrt(
                1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

            ax.plot(x, y_est, '-')
            ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
            ax.plot(x, y, 'o', color='tab:brown')
        return fig
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return fig


# Функция получения размеров полотна графика по ключевому слову
def get_plot_size(plot_size):
    width = 22
    height = 16
    if plot_size == "small":
        width = 16
        height = 10
    if plot_size == "medium":
        width = 22
        height = 16
    if plot_size == "large":
        width = 28
        height = 22
    return width, height


# Входная точка для построения графика визуализации результатов обучения модели (matplotlib)
def model_plot(model, data, model_columns, train_column):
    model_name = type(model).__name__
    if model_name == "DBSCAN":
        return v_dbscan.get_plot(model, data[model_columns])
    if model_name == "KMeans":
        return v_kmeans.get_plot(model, data[model_columns])
    if model_name == "Birch":
        return v_kmeans.get_plot(model, data[model_columns])
    if model_name == "AgglomerativeClustering":
        return v_aggcluster.get_plot(model, data[model_columns])
    if model_name == "RandomForestRegressor":
        return v_RFregressor.get_plot(model, data, model_columns, train_column)
    if model_name == "GaussianProcessRegressor":
        return v_GPregressor.get_plot(model, data, model_columns, train_column)
    if model_name == "LinearRegression":
        return v_Lregressor.get_plot_2(model, data, model_columns, train_column)
    return None


# Входная точка для построения графика визуализации результатов обучения модели (plotly)
def model_plotly(model, data, model_columns, train_column):
    model_name = type(model).__name__
    if model_name == "DBSCAN":
        return v_dbscan.get_plot_2(model, data, model_columns)
    if model_name == "KMeans":
        return v_kmeans.get_plot(model, data[model_columns])
    if model_name == "Birch":
        return v_kmeans.get_plot(model, data[model_columns])
    if model_name == "AgglomerativeClustering":
        return v_aggcluster.get_plot(model, data[model_columns])
    if model_name == "RandomForestRegressor":
        return v_RFregressor.get_plot_2(model, data, model_columns, train_column)
    if model_name == "GaussianProcessRegressor":
        return v_GPregressor.get_plot_2(model, data, model_columns, train_column)
    if model_name == "LinearRegression":
        return v_Lregressor.get_plot_2(model, data, model_columns, train_column)
    return None


# used for moving average (instead bottleneck)
def move_mean(y_pred_plot,moving_average_window,min_periods):
    numbers_series = pd.Series(y_pred_plot)
    windows = numbers_series.rolling(moving_average_window, min_periods=min_periods)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    return moving_averages_list


def get_plot_normal_values(data, draw, plot_size):
    data1 = data[300:700]
    data2 = data[300:700]
    moving_average_window = 5
    width, height = get_plot_size(plot_size)
    fig, ax = plt.subplots(figsize=(width, height))
    # plt.grid(True)

    x_label = data1.columns.values[0]
    y_label = data1.columns.values[3]
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.plot(data2[x_label], data2[y_label], "", color="tab:blue")

    y1 = move_mean(data1[y_label] * 1.05, moving_average_window, 1)
    y2 = move_mean(data1[y_label] * 0.95, moving_average_window, 1)
    plt.plot(data2[x_label], y1, draw, lw=1.5, color='tab:red', alpha=0.75)
    plt.plot(data2[x_label], y2, draw, lw=1.5, color='tab:red', alpha=0.75)

    return fig
