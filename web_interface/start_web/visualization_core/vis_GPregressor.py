import numpy as np

from matplotlib import pyplot as plt


def get_plot(model, data):
    data=data[:3000] # костыль
    time = data.loc[:, '_DATE_'].values
    data = data.drop(['_DATE_', '_NumMotor_'], axis=1)
    X = data.drop(['_MOT_TEMP_', '_VEL_AXIS_'], axis=1).values
    X = np.atleast_2d(X)
    x = X
    y = data.loc[:, '_MOT_TEMP_'].values

    y_pred, sigma = model.predict(x, return_std=True)

    fig = plt.figure(figsize=(30, 20))
    plt.plot(time, y, 'ro', markersize=10, label='Observations')
    plt.plot(time, y_pred, 'bo', markersize=5, label='Prediction')
    plt.legend(loc='upper left')
    return fig
