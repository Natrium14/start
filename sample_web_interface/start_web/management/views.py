from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    context = {
        'manage': 'manage'
    }
    return render(request, "management/index.html", context)

