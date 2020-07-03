import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_plot(model, data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    fig, ax = plt.subplots(figsize=(20, 18))
    y_pred = model.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=y_pred, cmap='Paired')
    plt.title(type(model).__name__)
    plt.legend()
    return fig
