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


def get_plot(model, data):
    data = data.drop(['_DATE_', '_NumMotor_'], axis=1)

    choosen_example_testrun = 70
    profile_id = choosen_example_testrun
    output_value = '_MOT_TEMP_'
    moving_average_window = 20

    #X_plot = data.drop(['_CURR_ACT_', '_VEL_AXIS_', '_MOT_TEMP_'], axis=1).loc[data['_VEL_AXIS_'] == profile_id].values
    X_plot = data.drop(['_MOT_TEMP_'], axis=1).loc[data['_VEL_AXIS_'] == profile_id].values
    y_plot = data.loc[data['_VEL_AXIS_'] == profile_id, output_value].values
    y_pred_plot = model.predict(X_plot)
    y_pred_plot_smooth = move_mean(y_pred_plot, moving_average_window, 1)
    time = np.linspace(0, y_plot.shape[0], num=y_plot.shape[0])

    with sns.axes_style("whitegrid"):
        fig = plt.figure(figsize=(30, 20))

        ax1 = fig.add_subplot(211)
        ax1.plot(time, y_pred_plot, label='predict without smoothing', color='red', alpha=0.4, linewidth=0.8)
        ax1.plot(time, y_plot, label='True', color='black', linewidth=1)
        ax1.legend(loc='best')
        ax1.set_title("profile id: {0} without smoothing".format(profile_id))

        ax2 = fig.add_subplot(212)
        ax2.plot(time, y_pred_plot_smooth, label='predict with smoothing', color='red', alpha=0.8, linewidth=0.8)
        ax2.plot(time, y_plot, label='True', color='black', linewidth=1)
        ax2.legend(loc='best')
        ax2.set_title("profile id: {0} with smoothing".format(profile_id))

    return fig
