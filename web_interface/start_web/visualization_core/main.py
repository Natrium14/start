import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vispy import plot as vp

import plotly.express as px
import plotly.graph_objects as go

import numpy as np

import io
import urllib, base64



def get_plot(data, draw, plot_size):
    width, height = get_plot_size(plot_size)
    fig, ax = plt.subplots(figsize=(width, height))
    #plt.grid(True)
    columns_count = len(data.columns)

    if columns_count < 2:
        return fig

    if columns_count == 2:
        x_label = data.columns.values[0]
        y_label = data.columns.values[1]
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.plot(data[x_label], data[y_label], draw)

    if columns_count > 2:
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        x_label = data.columns.values[0]
        ax.set_xlabel(x_label)
        mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:grey', 'tab:cyan']
        columns = data.columns[1:columns_count]
        date_column = data.columns[0:1][0]
        for i, column in enumerate(columns):
            label = ''.join(filter(whitelist.__contains__, column))
            plt.plot(data[date_column].values, data[column].values, draw, lw=1.5, color=mycolors[i], label=label)
        plt.legend(loc='upper left')

    return fig


def get_plot1(data, draw, plot_size):
    #fig = px.line(data, x=data.columns.values[0], y=data.columns.values[1], title='Зависимость 1')
    #fig.show()

    fig = go.Figure()
    columns_count = len(data.columns)

    if columns_count < 2:
        return fig

    if columns_count == 2:
        x_label = data.columns.values[0]
        y_label = data.columns.values[1]
        print("x")
        fig.add_trace(go.Scatter(x=data[x_label], y=data[y_label],
                                 mode='lines',
                                 name='lines'))
        #fig.show()
        return fig

    if columns_count > 2:
        columns = data.columns[1:columns_count]
        date_column = data.columns[0:1][0]
        for i, column in enumerate(columns):
            fig.add_trace(go.Scatter(x=data[date_column].values, y=data[column].values,
                                     mode='lines',
                                     name='lines'))
        return fig


def get_hist(data, bins, plot_size):
    width, height = get_plot_size(plot_size)
    fig, ax = plt.subplots(figsize=(width, height))
    try:
        plt.grid(True)
        x_label = data.columns.values[0]
        ax.set_xlabel(x_label)
        plt.hist(data[x_label], bins=bins, facecolor='blue', alpha=0.75)
        return fig
    except:
        return fig


def get_heatmap(data, plot_size):
    width, height = get_plot_size(plot_size)
    fig, ax = plt.subplots(figsize=(width, height))
    try:
        sns_plot = sns.heatmap(data.corr(), xticklabels=data.corr().columns,
                               yticklabels=data.corr().columns, cmap='RdYlGn', center=0, annot=True, square=True,
                               linewidths=.5, ax=ax, robust=True)
        plt.title('Корреляция', fontsize=20)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        fig = sns_plot.get_figure()
        return fig
    except:
        return fig


def get_fill_between(data, plot_size):
    width, height = get_plot_size(plot_size)
    fig, ax = plt.subplots(figsize=(width, height))
    plt.grid(True)
    columns_count = len(data.columns)
    try:
        if columns_count == 2:
            x_label = data.columns.values[0]
            y_label = data.columns.values[1]
            x = data[x_label].astype('int32')
            y = data[y_label]
            print(x.describe())
            print(y.describe())
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            a, b = np.polyfit(x, y, 1)
            y_est = a * x + b
            y_err = x.std() * np.sqrt(
                1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

            ax.plot(x, y_est, '-')
            ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
            ax.plot(x, y, 'o', color='tab:brown')
        return fig
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return fig


def get_plot_size(plot_size):
    width = 22
    height = 16
    if plot_size == "small":
        width = 16
        height = 10
    if plot_size == "medium":
        width = 22
        height = 16
    if plot_size == "large":
        width = 28
        height = 22
    return width, height
