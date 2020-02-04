import io
import time
import pandas as pd
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt

import sample_web_interface.start_web.ml_core.main as ml_core
import sample_web_interface.start_web.statistic_core.main as stat_core
import sample_web_interface.start_web.visualization_core.main as vis_core


data = None


# Стартовая страница для получения выборки и обучения модели
def index(request):
    if data is not None:
        context = {
            'dataset_description': str(data.count())
        }
        return render(request, "data_set/model_training.html", context)
    else:
        return render(request, "data_set/model_training.html")


# Метод получения таблицы статистических показателей выборки данных
def stat_index(request):
    min = stat_core.get_min(data['current_stator'])
    mean = stat_core.get_mean(data['current_stator'])
    max = stat_core.get_max(data['current_stator'])
    median = stat_core.get_median(data['current_stator'])
    variance = stat_core.get_variance(data['current_stator'])
    stdev = stat_core.get_stdev(data['current_stator'])

    context = {
        'min': min,
        'mean': mean,
        'max': max,
        'median': median,
        'variance': variance,
        'stdev': stdev
    }
    return render(request, "data_set/statistic.html", context)


# Метод получения выборки данных из файла
def upload_data(request):
    if request.POST:
        file = request.FILES.get('data_file')
        global data

        data = pd.read_csv(file, names=['time', 'temp_env', 'temp_stator', 'current_stator', 'freq', 'load', 'state'])
        context = {
            'dataset': data.head(),
            'dataset_description': str(data.count())
        }
        print(data.head())
        return render(request, "data_set/model_training.html", context)
    else:
        return render(request, "data_set/model_training.html")


# Вывод полученной выборки данных в отдельный html файл
def show_dataset(request):
    return HttpResponse(data.to_html())


def model_test(request):
    return render(request, "data_set/model_test.html")


# Метод получения графика зависимости одного атрибута от другого
def make_plot(request):
    fig = vis_core.get_plot(data['time'], data['current_stator'])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


# Метод обучения модели по выборке
def model_train(request):
    method = str(request.POST['methodSelect'])
    context = { }
    if method:
        timestamp1 = int(time.time())
        model = ml_core.model_train(data, method)
        timestamp2 = int(time.time())
        context['time_to_train'] = str(timestamp2-timestamp1)
        context['model_description'] = str(model)
    else:
        context['model_description'] = 'error'
    if data is not None:
        context['dataset_description'] = str(data.count())
    return render(request, "data_set/model_training.html", context)