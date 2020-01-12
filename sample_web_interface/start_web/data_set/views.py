from django.shortcuts import render
from django.http import HttpResponse
import sample_web_interface.start_web.ml_core.main as ml_core


def index(request):
    context = {
        'data': ml_core.hello_world()
    }
    return render(request, "data_set/model_training.html", context)


def model_test(request):
    return render(request, "data_set/model_test.html")

