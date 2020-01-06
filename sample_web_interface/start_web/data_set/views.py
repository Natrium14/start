from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return render(request, "data_set/model_training.html")


def model_test(request):
    return render(request, "data_set/model_test.html")

