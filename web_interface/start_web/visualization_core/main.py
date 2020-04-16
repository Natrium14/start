import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import io
import urllib, base64


def get_plot(data):
    fig, ax = plt.subplots(figsize=(20,18))
    columns_count = len(data.columns)

    if columns_count < 2:
        return None

    if columns_count == 2:
        x_label = data.columns.values[0]
        y_label = data.columns.values[1]
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.plot(data[x_label], data[y_label], 'bo')

    if columns_count > 2:
        mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:grey', 'tab:cyan']
        columns = data.columns[1:columns_count]
        for i, column in enumerate(columns):
            plt.plot(data["Time"].values, data[column].values, 'bo', lw=1.5, color=mycolors[i], label=column)
        plt.legend(loc='upper left')

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