import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# used for moving average (instead bottleneck)
def move_mean(y_pred_plot,moving_average_window,min_periods):
    numbers_series = pd.Series(y_pred_plot)
    windows = numbers_series.rolling(moving_average_window, min_periods=min_periods)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    return moving_averages_list


def get_plot(model, data, model_columns, train_column):
    X = data.loc[:, model_columns].values
    y = data.loc[:, train_column].values
    moving_average_window = 20

    y_pred_plot = model.predict(X)
    y_pred_plot_smooth = move_mean(y_pred_plot, moving_average_window, 1)
    time = np.linspace(0, y.shape[0], num=y.shape[0])

    with sns.axes_style("whitegrid"):
        fig = plt.figure(figsize=(30, 20))

        ax1 = fig.add_subplot(211)
        ax1.plot(time, y_pred_plot, label='predict without smoothing', color='red', alpha=0.4, linewidth=0.8)
        ax1.plot(time, y, label='True', color='black', linewidth=1)
        ax1.legend(loc='best')
        ax1.set_title("without smoothing")

        ax2 = fig.add_subplot(212)
        ax2.plot(time, y_pred_plot_smooth, label='predict with smoothing', color='red', alpha=0.8, linewidth=0.8)
        #ax2.plot(time, y, label='True', color='black', linewidth=1)
        ax2.plot(time[:3500], y[:3500], label='True', color='black', linewidth=1)
        ax2.plot(time[3500:4500], y[3500:4500], color='black', linewidth=1, alpha=0.0)
        ax2.plot(time[4500:6500], y[4500:6500], color='black', linewidth=1)
        ax2.plot(time[6500:], y[6500:], color='black', linewidth=1, alpha=0.0)
        ax2.legend(loc='best')
        ax2.set_title("with smoothing")

    return fig
