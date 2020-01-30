import io
import pandas as pd
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt

import sample_web_interface.start_web.ml_core.main as ml_core
import sample_web_interface.start_web.statistic_core.main as stat_core
import sample_web_interface.start_web.visualization_core.main as vis_core

data = None

def index(request):
    return render(request, "data_set/model_training.html")


# Метод получения таблицы статистических показателей выборки данных
def stat_index(request):
    if request.POST:
        global data

        min = stat_core.get_mean(data['current_stator'])
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
    else:
        return render(request, "data_set/statistic.html")


# Метод получения выборки данных из файла
def upload_data(request):
    if request.POST:
        file = request.FILES.get('data_file')
        global data
        data = pd.read_csv(file, names=['time', 'temp_env', 'temp_stator', 'current_stator', 'freq', 'load'])
        dataset_description = 'Количество записей: ' + str(data.count())
        data.to_html(classes='my_class')
        #data_table = data.head().to_html()
        context = {
            'dataset': data.head(),
            'dataset_description': dataset_description
        }
        return render(request, "data_set/model_training.html", context)
    else:
        return render(request, "data_set/model_training.html")


# Вывод полученной выборки данных в отдельный html файл
def show_dataset(request):
    return HttpResponse(data.to_html())


def model_test(request):
    return render(request, "data_set/model_test.html")


def make_plot(request):
    fig = vis_core.get_plot([],[])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response

