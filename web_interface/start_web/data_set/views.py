import io
import time
import json
import pandas as pd
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
        context = {
            'dataset_description': data.columns
        }
        return render(request, "data_set/model_training.html", context)
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
    method = str(request.POST['methodSelect'])
    columns = request.POST.getlist('model_columns')

    context = { }
    try:
        if method:
            try:
                global model
                global data_train

                data_train = data[columns]
                timestamp1 = int(time.time())
                model = ml_core.model_train(data_train, method)
                print(set(model.labels_))
                timestamp2 = int(time.time())
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