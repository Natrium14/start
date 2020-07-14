import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_plot(model, data):
    fig, ax = plt.subplots(figsize=(20, 18))
    try:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        y_pred = model.fit_predict(data)
        plt.scatter(data[:, 0], data[:, 1], c=y_pred, cmap='Paired', edgecolors="black")
        plt.title(type(model).__name__)
        plt.legend()
        return fig
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return fig