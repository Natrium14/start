from django.shortcuts import render
from django.http import HttpResponse


def contact(request):
    return render(request, "description/contact.html")


def about(request):
    return render(request, "description/about.html")