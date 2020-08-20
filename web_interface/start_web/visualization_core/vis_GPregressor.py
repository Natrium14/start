import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from matplotlib import pyplot as plt
import seaborn as sns


# used for moving average (instead bottleneck)
def move_mean(y_pred_plot,moving_average_window,min_periods):
    numbers_series = pd.Series(y_pred_plot)
    windows = numbers_series.rolling(moving_average_window, min_periods=min_periods)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    return moving_averages_list


def get_plot(model, data, model_columns, train_column):
    moving_average_window = 20

    X = data.loc[:, model_columns].values
    X = np.atleast_2d(X)
    x = X
    y = data.loc[:, train_column].values
    time = np.linspace(0, y.shape[0], num=y.shape[0])
    y_pred, sigma = model.predict(x, return_std=True)

    y_pred_plot_smooth = move_mean(y_pred, moving_average_window, 1)

    with sns.axes_style("whitegrid"):
        fig = plt.figure(figsize=(30, 20))

        ax1 = fig.add_subplot(211)
        ax1.plot(time, y_pred, 'bo', label='prediction', color='red', alpha=0.4, markersize=10)
        ax1.plot(time, y, 'ro', label='True', color='black', markersize=5)
        #ax1.plot(time[:3000], y[:3000], 'ro', label='True', color='black', markersize=5)
        #ax1.plot(time[3000:], y[3000:], 'ro', label='True', color='black', alpha=0.0, markersize=5)
        ax1.legend(loc='best')
        ax1.set_title("Прогнозирование температуры")

        ax2 = fig.add_subplot(212)
        ax2.plot(time, y_pred_plot_smooth, 'bo', label='prediction', color='red', alpha=0.4, markersize=10)
        ax2.plot(time, y, 'ro', label='True', color='black', markersize=5)
        #ax2.plot(time[:3000], y[:3000], 'ro', label='True', color='black', markersize=5)
        #ax2.plot(time[3000:], y[3000:], 'ro', label='True', color='black', alpha=0.0,markersize=5)
        ax2.legend(loc='best')
        ax2.set_title("Прогнозирование температуры со сглаживанием")

        #plt.plot(time, y, 'ro', markersize=10, label='Observations')
        #plt.plot(time, y_pred, 'bo', markersize=5, label='Prediction')
        #plt.legend(loc='upper left')
    return fig


def get_plot_2(model, data, model_columns, train_column):
    X = data.loc[:, model_columns].values
    X = np.atleast_2d(X)
    time = data.iloc[:, 0].values.reshape(-1)
    y_pred = model.predict(X)

    fig = px.scatter(data, x=data.columns[0], y=train_column, opacity=0.3)
    fig.add_traces(go.Scatter(x=time, y=y_pred, name=train_column + " prediction", mode='markers'))
    return fig
