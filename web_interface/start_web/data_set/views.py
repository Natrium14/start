import io
import time
import pandas as pd
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt

import ml_core.main as ml_core
import statistic_core.main as stat_core
import visualization_core.main as vis_core


data = None


# Стартовая страница
def main_page(request):
    return render(request, "data_set/main_page.html")

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
    try:
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

            return render(request, "data_set/model_training.html", context)
        except Exception:
            return render(request, "error/error404.html")
    else:
        return render(request, "data_set/model_training.html")


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
    context = { }
    try:
        if method:
            try:
                timestamp1 = int(time.time())
                model = ml_core.model_train(data, method)
                timestamp2 = int(time.time())
                context['time_to_train'] = str(timestamp2-timestamp1)
                context['model_description'] = str(model)
            except Exception:
                return render(request, "error/error404.html")
        else:
            context['model_description'] = 'error'
        if data is not None:
            context['dataset_description'] = str(data.count())
        return render(request, "data_set/model_training.html", context)
    except Exception:
        return render(request, "error/error404.html")