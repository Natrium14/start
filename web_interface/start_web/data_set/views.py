import datetime
import io
import time
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from joblib import dump, load
from tkinter import *
from tkinter import filedialog
from plotly import io as plotly_io


import ml_core.main as ml_core
import statistic_core.main as stat_core
import visualization_core.main as vis_core
import management.views as m_views
import data_set.data_entity as data11

data = None

# Стартовая страница
def main_page(request):
    return render(request, "data_set/main_page.html")


# Стартовая страница для получения выборки
def index(request):
    context = {}
    timestamp1 = int(time.time())

    global data
    if data is None:
        data = data11.data()

    data.set_db_client(m_views.get_client())
    print(data.get_db_client())
    context['connection'] = data.get_db_client()

    if data.get_data() is not None:
        context['dataset_count'] = len(data.get_data()[[data.get_data().columns[0]]])
        context['dataset_description'] = data.get_data().columns

    timestamp2 = int(time.time())
    print("Загрузка страницы data_set/index_dataset: ", timestamp2-timestamp1, " секунд")
    return render(request, "data_set/index_dataset.html", context)


# Метод получения таблицы статистических показателей выборки данных
def stat_index(request):
    global data

    try:
        array = {}
        timestamp1 = int(time.time())
        data_current = data.get_data()
        for col in data_current.columns:
            column = str(col)
            try:

                #if "cur" in column.lower():
                if True:
                    min = stat_core.get_min(data_current[column])
                    mean = stat_core.get_mean(data_current[column])
                    max = stat_core.get_max(data_current[column])
                    median = stat_core.get_median(data_current[column])
                    variance = stat_core.get_variance(data_current[column])
                    stdev = stat_core.get_stdev(data_current[column])
                    array[column] = {
                        'min': min,
                        'mean': mean,
                        'max': max,
                        'median': median,
                        'variance': variance,
                        'stdev': stdev
                    }
            except:
                pass
        timestamp2 = int(time.time())
        print("Загрузка страницы data_set/statistic:", timestamp2 - timestamp1, " секунд")
        context = {
            'array': array
        }
        return render(request, "data_set/statistic.html", context)
    except Exception:
        error_context = {
            'error': sys.exc_info()[0],
        }
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html", error_context)


def normal_values(request):
    global data

    array = {}
    data_current = data.get_data()
    for col in data_current.columns:
        column = str(col)
        try:
            min = stat_core.get_min(data_current[column])
            max = stat_core.get_max(data_current[column])
            normal_value = str(round(min, 4)) + " .. " + str(round(max, 4)) + "; допуск: " + str(abs(round(max, 4)-round(min, 4))*0.05)

            array[column] = {
                "normal_value": normal_value
            }
        except:
            pass
    context = {
        'array': array
    }
    print(array)
    return render(request, "data_set/normal_values.html", context)


def vis_normal_values(request):
    global data

    data_current = data.get_data()
    fig = vis_core.get_plot_normal_values(data_current, "", "medium")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


# Метод получения выборки данных из файла
def upload_data(request):
    global data

    context = {}
    timestamp1 = int(time.time())
    if request.POST:
        try:
            file = request.FILES.get('data_file')

            data_current = pd.read_csv(file)
            dataset_count = len(data_current[[data_current.columns[0]]])
            dataset_description = data_current.columns

            # Костыль
            try:
                data_current[data_current.columns[0]] = data_current.index
                data_current = data_current.drop(['null'], axis=1)
                data_current = data_current.loc[data_current['_NumMotor_'] == "_1_"]
            except:
                pass

            data.set_data(data_current)

            context['connection'] = data.get_db_client()
            context['dataset_count'] = dataset_count
            context['dataset_description'] = dataset_description

            timestamp2 = int(time.time())
            print("Загрузка выборки из файла:", timestamp2 - timestamp1, " секунд")
            return render(request, "data_set/index_dataset.html", context)
        except Exception:
            error_context = {
                'error': str(sys.exc_info()[0]),
                'reason': "Загруженный файл имеет отличный от CSV формат. Формат данных не соответствует табличной форме представления. Файл битый."
            }
            print("Unexpected error:", sys.exc_info()[0])
            return render(request, "error/error404.html", error_context)
    else:
        return render(request, "data_set/index_dataset.html")


# Метод получения выборки из БД
def upload_data_db(request):
    global data

    context = {}
    timestamp1 = int(time.time())

    if request.POST:
        try:
            context['connection'] = data.get_db_client()
            db = data.get_db_client()['start']
            collection = db['sample']

            id = request.POST["id"]
            ot = int(request.POST["number_ot"])
            do = int(request.POST["number_do"])

            # [0] - для получения первого элемента
            data_current = pd.DataFrame(list(collection.find())[0]["data"])
            data_current = data_current.transpose()

            #data = pd.DataFrame(list(cursor))
            # if ot != -1 and do != -1 and do > ot:
            #     data = pd.DataFrame(list(cursor))[ot:do]
            # else:
            #     data = pd.DataFrame(list(cursor))
            #del data['_id']

            for col in data_current.columns[1:]:
                try:
                    data_current[col] = data_current[col].astype(float)
                except:
                    pass

            # Костыль
            try:
                data_current[data_current.columns[0]] = data_current.index
                data_current = data_current.drop(['null'], axis=1)
                data_current = data_current.loc[data_current['_NumMotor_'] == "_1_"]
            except:
                pass

            data.set_data(data_current)

            context['dataset_count'] = len(data.get_data()[[data.get_data().columns[0]]])
            context['dataset_description'] = data.get_data().columns

            timestamp2 = int(time.time())
            print("Загрузка выборки из БД:", timestamp2 - timestamp1, " секунд")
            return redirect('/data_set/index', context)
        except Exception:
            error_context = {
                'error': str(sys.exc_info()[0]),
                'reason': "Отсутствует подключение к БД. Не найдено искомой таблицы/коллекции. Не найдено информации по заданному ID"
            }
            print("Unexpected error:", sys.exc_info()[0])
            return render(request, "error/error404.html", error_context)
    else:
        return render(request, "data_set/index_dataset.html")


# Страница с обучением моделей
def model_training(request):
    global data

    context = {}
    timestamp1 = int(time.time())
    try:
        if data.get_data() is not None:
            context['dataset_description'] = data.get_data().columns

        if data.get_model() is not None:
            context['model_description'] = str(data.get_model())

        if data.get_db_client() is not None:
            context['connection'] = data.get_db_client()

        timestamp2 = int(time.time())
        print("Загрузка страницы data_set/model_train:", timestamp2 - timestamp1, " секунд")
        return render(request, "data_set/model_train.html", context)
    except Exception:
        error_context = {
            'error': str(sys.exc_info()[0])
        }
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html", error_context)


# Вывод полученной выборки данных в отдельный html файл
def show_dataset(request):
    global data

    if data.get_data() is not None:
        return HttpResponse(data.get_data().to_html())

    else:
        error_context = {
            'error': str(sys.exc_info()[0])
        }
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html", error_context)


# метод получения обученных моделей из бд
def model_test_page(request):
    global data

    context = {}
    timestamp1 = int(time.time())

    try:
        context['connection'] = data.get_db_client()
        db = data.get_db_client()['start']
        collection = db['models']

        # получение списка сохраненных моделей
        models = list(collection.find())
        model_view_array = []
        for m in models:
            model_view_dict = {}
            model_view_dict["id"] = m["_id"]
            model_view_dict["name"] = m["name"]
            model_view_dict["created_time"] = datetime.datetime.fromtimestamp(m["created_time"]).strftime('%Y-%m-%d %H:%M:%S')
            model_view_array.append(model_view_dict)
        context["models"] = model_view_array

        # получение сохраненных обучающих выборок
        collection = db['sample']
        samples = list(collection.find())
        sample_view_array = []
        for s in samples:
            sample_view_dict = {}
            sample_view_dict["id"] = s["_id"]
            sample_view_dict["count"] = len(s["data"])
            sample_view_array.append(sample_view_dict)
        context["samples"] = sample_view_array

        timestamp2 = int(time.time())
        print("Получение выборок и моделей из БД:", timestamp2 - timestamp1, " секунд")
        return render(request, "data_set/model_test.html", context)
    except:
        error_context = {
            'error': str(sys.exc_info()[0]),
            'reason': "Отсутствует подключение к БД. Не найдено искомой таблицы/коллекции."
        }
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html", error_context)


# Метод получения графика зависимости одного атрибута от другого
def make_plot(request):
    global data

    try:
        timestamp1 = int(time.time())

        params = {}
        params["ot"] = ot = int(request.POST['number_ot'])
        params["do"] = do = int(request.POST['number_do'])
        params["type"] = str(request.POST['type'])
        params["draw"] = str(request.POST['draw'])
        params["bins"] = int(request.POST['bins'])
        params["plot_size"] = str(request.POST['plot_size'])

        columns = request.POST.getlist('checkbox_columns')
        vis_data = data.get_data()[columns]
        if do > ot:
            vis_data = data.get_data()[columns][ot:do]

        fig = vis_core.data_plot(vis_data, params)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        response = HttpResponse(buf.getvalue(), content_type='image/png')

        timestamp2 = int(time.time())
        print("Построение графика:", timestamp2 - timestamp1, " секунд")
        return response
    except Exception:
        error_context = {
            'error': str(sys.exc_info()[0]),
            'reason': "Формат данных непозволяет построить данный график. Не выбраны необходимые поля."
        }
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html", error_context)


# Метод обучения модели по выборке
def model_train(request):
    global data

    method = str(request.POST['method_name'])

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
                if request.POST['birch_clusters']:
                    params["birch_clusters"] = int(request.POST['birch_clusters'])
                if request.POST['agg_clusters']:
                    params["agg_clusters"] = int(request.POST['agg_clusters'])

                if request.POST.getlist('model_columns'):
                    model_columns = request.POST.getlist('model_columns')
                    data.set_model_columns(model_columns)
                    params["model_columns"] = model_columns
                if request.POST.getlist('model_column_train'):
                    train_column = str(request.POST.getlist('model_column_train')[0])
                    data.set_train_column(train_column)
                    params["column_train"] = train_column
            except Exception:
                pass

            # create and train model
            try:
                timestamp1 = int(time.time())
                model = ml_core.model_train(data.get_data(), method, params)
                print(type(model).__name__)
                timestamp2 = int(time.time())
                data.set_model(model)
                print("Время обучения модели: ", str(timestamp2-timestamp1), " секунд")
                # save model to file
                '''
                try:
                    pass
                    #save_file = filedialog.asksaveasfile(mode='w')
                    #file_name = save_file.name
                    #dump(model, file_name + '.joblib')
                    #save_file.close()
                except Exception:
                    pass
                '''
                context['time_to_train'] = str(timestamp2-timestamp1)
                context['model_description'] = str(model)
            except:
                return render(request, "error/error404.html")
        else:
            context['model_description'] = None
            pass
        if data.get_data() is not None:
            context['dataset_description'] = data.get_data().columns

        context['connection'] = data.get_db_client()
        return render(request, "data_set/model_train.html", context)
    except:
        error_context = {
            'error': str(sys.exc_info()[0])
        }
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html", error_context)


# get abnormal values from dbscan
def get_anomalies(request):
    global data

    data_current = data.get_data()
    model = data.get_model()
    timestamp1 = int(time.time())
    try:
        abnormal_data = data_current.iloc[0]
        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True
        labels = model.labels_
        unique_labels = set(labels)
        counts = np.bincount(labels[labels >= 0])

        for k in unique_labels:
            class_member_mask = (labels == k)
            if k == -1:
                abnormal_data = data_current[class_member_mask & ~core_samples_mask]
        for k in unique_labels:
            class_member_mask = (labels == k)
            if counts[k] < (len(data_current[[data_current.columns[0]]])*0.1):
                abnormal_data = abnormal_data.append(data_current[class_member_mask & core_samples_mask], ignore_index=True)
        #abnormal_data = abnormal_data.sort_values(by="Time")

        timestamp2 = int(time.time())
        print("Получение аномальных значений: ", str(timestamp2 - timestamp1), " секунд")
        return HttpResponse(abnormal_data.to_html())
    except Exception:
        error_context = {
            'error': str(sys.exc_info()[0]),
            'reason': "Отсутствуют выявленные аномалии."
        }
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html", error_context)


# save model function
def save_model(request):
    global data

    context = {'connection': data.get_db_client()}
    timestamp1 = int(time.time())
    try:
        ml_core.model_save(data.get_model())
        context['dataset_description'] = data.get_data().columns
        timestamp2 = int(time.time())
        print("Сохранение модели в БД: ", str(timestamp2 - timestamp1), " секунд")
        return render(request, "data_set/model_train.html", context)
    except:
        error_context = {
            'error': str(sys.exc_info()[0]),
            'reason': "Отсутствует подключение к БД."
        }
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html", error_context)


# Метод визуализации
def vis_model(request):
    global data

    timestamp1 = int(time.time())

    fig = vis_core.model_plot(
        data.get_model(),
        data.get_data(),
        data.get_model_columns(),
        data.get_train_column()
    )
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)

    timestamp2 = int(time.time())
    print("Визуализация модели: ", str(timestamp2 - timestamp1), " секунд")

    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response
