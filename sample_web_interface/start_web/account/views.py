from django.shortcuts import render


# Create your views here.
def login(request):
    return render(request, "account/login.html")


def logout(request):
    return render(request, "data_set/model_training.html")


def register(request):
    return render(request, "account/register.html")


def cabinet(request):
    return render(request, "account/cabinet.html")


def change_password(request):
    return render(request, "data_set/model_training.html")


def change_info(request):
    return render(request, "data_set/model_training.html")