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


import ml_core.main as ml_core
import statistic_core.main as stat_core
import visualization_core.main as vis_core
import visualization_core.vis_dbscan as v_dbscan
import visualization_core.vis_kmeans as v_kmeans
import visualization_core.vis_aggcluster as v_aggcluster
import management.views as m_views


from sklearn.cluster import DBSCAN

data = None
data_train = None
model = None
db_client = None


# Стартовая страница
def main_page(request):
    return render(request, "data_set/main_page.html")


# Стартовая страница для получения выборки и обучения модели
def index(request):
    context = {}
    try:
        global db_client
        db_client = m_views.get_client()
        context['connection'] = db_client
    except:
        pass

    if data is not None:
        context['dataset_count'] = len(data[[data.columns[0]]])
        context['dataset_description'] = data.columns
    return render(request, "data_set/index_dataset.html", context)


# Метод получения таблицы статистических показателей выборки данных
def stat_index(request):
    try:
        array = {}
        timestamp1 = int(time.time())
        for col in data.columns:
            column = str(col)
            try:

                #if "cur" in column.lower():
                if True:
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

            except:
                pass
        timestamp2 = int(time.time())
        print("Стат. вычисления:", timestamp2 - timestamp1)
        context = {
            'array': array
        }
        return render(request, "data_set/statistic.html", context)
    except Exception:
        return render(request, "error/error404.html")


# Метод получения выборки данных из файла
def upload_data(request):
    context = {}
    if db_client is None:
        context['connection'] = db_client
    if request.POST:
        try:
            file = request.FILES.get('data_file')
            global data

            data = pd.read_csv(file)

            # Костыль
            try:
                data = data.loc[data['_NumMotor_'] == "_1_"]
                data['_DATE_'] = data.index
            except:
                pass

            #print(data.head())

            context['connection'] = db_client
            context['dataset_count'] = len(data[[data.columns[0]]])
            context['dataset_description'] = data.columns

            return render(request, "data_set/index_dataset.html", context)
        except Exception:
            return render(request, "error/error404.html")
    else:
        return render(request, "data_set/index_dataset.html")


# Метод получения выборки из БД
def upload_data_db(request):
    global db_client

    context = {}

    if db_client is None:
        print("0")
        return render(request, "error/error404.html")
    if request.POST:
        try:
            global data

            context['connection'] = db_client
            db = db_client['start']
            collection = db['sample']

            id = request.POST["id"]
            ot = int(request.POST["number_ot"])
            do = int(request.POST["number_do"])

            # [0] - для получения первого элемента
            data = pd.DataFrame(list(collection.find())[0]["data"])
            data = data.transpose()
            #print(data)
            #data = pd.DataFrame(list(cursor))
            # if ot != -1 and do != -1 and do > ot:
            #     data = pd.DataFrame(list(cursor))[ot:do]
            # else:
            #     data = pd.DataFrame(list(cursor))
            #del data['_id']

            for col in data.columns[1:]:
                try:
                    data[col] = data[col].astype(float)
                except:
                    pass

            # Костыль
            try:
                data = data.loc[data['_NumMotor_'] == "_1_"]
                data['_DATE_'] = data.index
            except:
                pass
            #print(len(data[[data.columns[0]]]))
            context['dataset_count'] = len(data[[data.columns[0]]])
            context['dataset_description'] = data.columns

            #return render(request, "data_set/index_dataset.html", context)
            return redirect('/data_set/index', context)
        except Exception:
            print("Unexpected error:", sys.exc_info()[0])
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
            return render(request, "data_set/model_train.html", context)
        else:
            return render(request, "data_set/model_train.html")
    except Exception:
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html")


# Вывод полученной выборки данных в отдельный html файл
def show_dataset(request):
    if data is not None:
        return HttpResponse(data.to_html())
    else:
        return render(request, "error/error404.html")


# метод получения обученных моделей из бд
def model_test_page(request):
    global db_client

    context = {}

    context['connection'] = db_client
    db = db_client['start']
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

    return render(request, "data_set/model_test.html", context)


# Метод получения графика зависимости одного атрибута от другого
def make_plot(request):
    try:
        ot = int(request.POST['number_ot'])
        do = int(request.POST['number_do'])
        type = str(request.POST['type'])
        draw = str(request.POST['draw'])
        bins = int(request.POST['bins'])
        plot_size = str(request.POST['plot_size'])
        #timestamp1 = int(time.time())

        columns = request.POST.getlist('checkbox_columns')
        vis_data = data[columns]
        fig = None
        if do > ot:
            vis_data = data[columns][ot:do]
        if type == "plot":
            fig = vis_core.get_plot(vis_data, draw, plot_size)
        if type == "hist":
            fig = vis_core.get_hist(vis_data, bins, plot_size)
        if type == "heatmap":
            fig = vis_core.get_heatmap(vis_data, plot_size)
        if type == "fill_between":
            fig = vis_core.get_fill_between(vis_data, plot_size)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        response = HttpResponse(buf.getvalue(), content_type='image/png')
        #timestamp2 = int(time.time())
        #print(timestamp2 - timestamp1)
        return response
    except Exception:
        return render(request, "error/error404.html")


# Метод обучения модели по выборке
def model_train(request):
    global db_client

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
                if request.POST['birch_clusters']:
                    params["birch_clusters"] = int(request.POST['birch_clusters'])
                if request.POST['agg_clusters']:
                    params["agg_clusters"] = int(request.POST['agg_clusters'])

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
                print("Время обучения:" + str(timestamp2-timestamp1))
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
                #context['time_to_train'] = str(timestamp2-timestamp1)
                #context['model_description'] = str(model)
            except:
                return render(request, "error/error404.html")
        else:
            context['model_description'] = None
            pass
        if data is not None:
            context['dataset_description'] = data.columns
            pass
        context['connection'] = db_client
        return render(request, "data_set/model_train.html", context)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html")


# get abnormal values from dbscan
def get_anomalies(request):
    try:
        abnormal_data = data.iloc[0]
        print('1')
        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True
        labels = model.labels_
        unique_labels = set(labels)
        print('1')
        counts = np.bincount(labels[labels >= 0])

        for k in unique_labels:
            class_member_mask = (labels == k)
            if k == -1:
                abnormal_data = data[class_member_mask & ~core_samples_mask]
        print('1')
        for k in unique_labels:
            class_member_mask = (labels == k)
            if counts[k] < (len(data[[data.columns[0]]])*0.1):
                abnormal_data = abnormal_data.append(data[class_member_mask & core_samples_mask], ignore_index=True)
        print('1')
        #abnormal_data = abnormal_data.sort_values(by="Time")

        return HttpResponse(abnormal_data.to_html())
    except Exception:
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html")


# save model function
def save_model(request):
    global db_client
    global model

    context = {'connection': db_client}
    try:
        print("1")
        ml_core.model_save(model)
        print("2")
        context['dataset_description'] = data.columns
        return render(request, "data_set/model_train.html", context)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return render(request, "error/error404.html")


# Метод визуализации
def vis_model(request):
    try:
        global model
        model_name = type(model).__name__
        if model_name == "DBSCAN":
            try:
                fig = v_dbscan.get_plot(model, data_train)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                response = HttpResponse(buf.getvalue(), content_type='image/png')
                return response
            except Exception:
                return render(request, "error/error404.html")
            #return vis_dbscan(request)
        if model_name == "KMeans":
            try:
                fig = v_kmeans.get_plot(model, data_train)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                response = HttpResponse(buf.getvalue(), content_type='image/png')
                return response
            except Exception:
                return render(request, "error/error404.html")
        if model_name == "Birch":
            try:
                fig = v_kmeans.get_plot(model, data_train)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                response = HttpResponse(buf.getvalue(), content_type='image/png')
                return response
            except Exception:
                return render(request, "error/error404.html")
        if model_name == "AgglomerativeClustering":
            try:
                fig = v_aggcluster.get_plot(model, data_train)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                response = HttpResponse(buf.getvalue(), content_type='image/png')
                return response
            except Exception:
                return render(request, "error/error404.html")
            #return vis_kmeans(request)
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