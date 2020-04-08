import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import io
import urllib, base64


def get_plot(data_x, data_y):
    fig, ax = plt.subplots()
    ax.set_xlabel('Время')
    ax.set_ylabel('Ток')
    plt.plot(data_x, data_y, 'bo')
    return fig


def get_chart(x,y,type):
    try:
        fig, ax = plt.subplots()
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if type == "plot":
            plt.plot(x, y, 'bo')
            return fig
        if type == "hist":
            plt.hist(x, y, bins=len(x)/7)
            return fig
        if type == "scatter":
            plt.scatter(x, y)
            return fig
    except Exception:
        fig, ax = plt.subplots()
        ax.set_xlabel('Error')
        plt.plot([1], [1], 'bo')
        return fig


#
def get_chart_corr(dataframe):
    try:
        if dataframe:
            fig, ax = plt.subplots()
            plt.figure(figsize=(11,10), dpi=85)
            sns_plot = sns.heatmap(dataframe.corr(), xticklabels=dataframe.corr().columns, yticklabels=dataframe.corr().columns, cmap='RdYlGn', center=0, annot=True)
            plt.title('Корреляция', fontsize=20)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            fig = sns_plot.get_figure()
            return fig
    except Exception:
        fig, ax = plt.subplots()
        ax.set_xlabel('Error')
        plt.plot([1], [1], 'bo')
        return fig