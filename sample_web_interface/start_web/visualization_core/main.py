import pandas as pd
import matplotlib.pyplot as plt

import io
import urllib, base64


def get_plot(data_x, data_y):
    fig, ax = plt.subplots()
    ax.set_xlabel('Время')
    ax.set_ylabel('Ток')
    plt.plot(data_x, data_y, 'bo')
    return fig
