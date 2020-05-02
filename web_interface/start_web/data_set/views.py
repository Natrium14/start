import io
import time
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from joblib import dump, load
from tkinter import *
from tkinter import filedialog


import ml_core.main as ml_core
import statistic_core.main as stat_core
import visualization_core.main as vis_core
import visualization_core.vis_dbscan as v_dbscan
import visualization_core.vis_kmeans as v_kmeans


from sklearn.cluster import DBSCAN

data = None
data_train = None
model = None


# Стартовая страница
def main_page(request):
    return render(request, "data_set/main_page.html")


# Стартовая страница для получения выборки и обучения модели
def index(request):
    if data is not None:
        context = {
            'dataset_count': len(data[[data.columns[0]]]),
            'dataset_description': data.columns
        }
        return render(request, "data_set/index_dataset.html", context)
    else:
        return render(request, "data_set/index_dataset.html")


# Метод получения таблицы статистических показателей выборки данных
def stat_index(request):
    try:
        array = {}

        for col in data.columns:
            column = str(col)
            if "current" in column.lower():
                min = stat_core.get_min(data[column])
                mean = stat_core.get_mean(data[column])
                max = stat_core.get_max(data[column])
                median = stat_core.get_median(data[column])
                variance = stat_core.get_variance(data[column])
                stdev = stat_core.get_stdev(data[column])
                array[column] = {
                    'min': min,
                    'mean': mean,
                    'max': max,
                    'median': median,
                    'variance': variance,
                    'stdev': stdev
                }

        context = {
            'array': array
        }
        return render(request, "data_set/statistic.html", context)
    except Exception:
        return render(request, "error/error404.html")


# Метод получения выборки данных из файла
def upload_data(request):
    if request.POST:
        try:
            file = request.FILES.get('data_file')
            global data

            data = pd.read_csv(file)

            context = {
                'dataset_count': len(data[[data.columns[0]]]),
                'dataset_description': data.columns
            }

            return render(request, "data_set/index_dataset.html", context)
        except Exception:
            return render(request, "error/error404.html")
    else:
        return render(request, "data_set/index_dataset.html")


# Страница с обучением моделей
def model_training(request):
    try:
        if data is not None:
            context = {
                'dataset_description': data.columns
            }
            return render(request, "data_set/model_training.html", context)
        else:
            return render(request, "data_set/model_training.html")
    except Exception:
        return render(request, "error/error404.html")


# Вывод полученной выборки данных в отдельный html файл
def show_dataset(request):
    if data is not None:
        return HttpResponse(data.to_html())
    else:
        return render(request, "error/error404.html")


def model_test(request):
    return render(request, "data_set/model_test.html")


# Метод получения графика зависимости одного атрибута от другого
def make_plot(request):
    try:
        columns = request.POST.getlist('checkbox_columns')
        vis_data = data[columns]
        fig = vis_core.get_plot(vis_data)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        response = HttpResponse(buf.getvalue(), content_type='image/png')
        return response
    except Exception:
        return render(request, "error/error404.html")


# Метод обучения модели по выборке
def model_train(request):
    method = str(request.POST['method_name'])
    columns = request.POST.getlist('model_columns')

    params = {}
    context = {}
    try:
        if method:
            # method parameters
            try:
                if request.POST['eps']:
                    params["eps"] = float(request.POST['eps'])
                if request.POST['min_samples']:
                    params["min_samples"] = int(request.POST['min_samples'])
                if request.POST['n_clusters']:
                    params["n_clusters"] = int(request.POST['n_clusters'])
                if request.POST['n_init']:
                    params["n_init"] = int(request.POST['n_init'])

            except Exception:
                pass
            # create and train model
            try:
                global model
                global data_train

                data_train = data[columns]
                timestamp1 = int(time.time())
                model = ml_core.model_train(data_train, method, params)
                timestamp2 = int(time.time())
                print(set(model.labels_))
                print(type(model).__name__)
                # save model to file
                try:
                    pass
                    #save_file = filedialog.asksaveasfile(mode='w')
                    #file_name = save_file.name
                    #dump(model, file_name + '.joblib')
                    #save_file.close()
                except Exception:
                    pass

                context['time_to_train'] = str(timestamp2-timestamp1)
                context['model_description'] = str(model)
            except Exception:
                return render(request, "error/error404.html")
        else:
            context['model_description'] = None
        if data is not None:
            context['dataset_description'] = data.columns
        return render(request, "data_set/model_training.html", context)
    except Exception:
        return render(request, "error/error404.html")


# get abnormal values from dbscan
def get_anomalies(request):
    try:
        abnormal_data = data.iloc[0]

        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True
        labels = model.labels_
        unique_labels = set(labels)

        counts = np.bincount(labels[labels >= 0])

        for k in unique_labels:
            class_member_mask = (labels == k)
            if k == -1:
                abnormal_data = data[class_member_mask & ~core_samples_mask]

        for k in unique_labels:
            class_member_mask = (labels == k)
            if counts[k] < (len(data[[data.columns[0]]])*0.1):
                abnormal_data = abnormal_data.append(data[class_member_mask & core_samples_mask], ignore_index=True)

        abnormal_data = abnormal_data.sort_values(by="Time")

        return HttpResponse(abnormal_data.to_html())
    except Exception:
        return render(request, "error/error404.html")


# Метод визуализации
def vis_model(request):
    try:
        global model
        model_name = type(model).__name__
        if model_name == "DBSCAN":
            return vis_dbscan(request)
        if model_name == "KMeans":
            return vis_kmeans(request)
    except Exception:
        return render(request, "error/error404.html")


# Визуализация dbscan
def vis_dbscan(request):
    try:
        fig = v_dbscan.get_plot(model, data_train)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        response = HttpResponse(buf.getvalue(), content_type='image/png')
        return response
    except Exception:
        return render(request, "error/error404.html")


# Визуализация kmeans
def vis_kmeans(request):
    try:
        fig = v_kmeans.get_plot(model, data_train)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        response = HttpResponse(buf.getvalue(), content_type='image/png')
        return response
    except Exception:
        return render(request, "error/error404.html")