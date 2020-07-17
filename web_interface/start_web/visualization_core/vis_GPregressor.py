import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

def get_plot(model, data):
    data=data[:3000] # костыль
    time = data.loc[:, '_DATE_'].values
    data = data.drop(['_DATE_', '_NumMotor_'], axis=1)
    X = data.drop(['_MOT_TEMP_', '_VEL_AXIS_'], axis=1).values
    X = np.atleast_2d(X)
    x = X
    y = data.loc[:, '_MOT_TEMP_'].values

    time = np.linspace(0, y.shape[0], num=y.shape[0])

    y_pred, sigma = model.predict(x, return_std=True)

    with sns.axes_style("whitegrid"):
        fig = plt.figure(figsize=(20, 10))

        ax1 = fig.add_subplot(111)
        ax1.plot(time, y_pred, 'bo', label='prediction', color='red', alpha=0.4, markersize=10)
        ax1.plot(time, y, 'ro', label='True', color='black', markersize=5)
        ax1.legend(loc='best')
        ax1.set_title("Прогнозирование температуры")

        #plt.plot(time, y, 'ro', markersize=10, label='Observations')
        #plt.plot(time, y_pred, 'bo', markersize=5, label='Prediction')
        #plt.legend(loc='upper left')
    return fig
