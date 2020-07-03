from pymongo import MongoClient

from django.shortcuts import render, redirect
from django.http import HttpResponse


# MongoClient
client = None


def index(request):
    context = {
        "connection": client
    }
    return render(request, "management/index.html", context)


def create_connection(request):
    if request.POST:
        try:
            global client

            server_address = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false'
            server_port = 27017
            client = MongoClient(server_address)
            #db = client['start']
            #collection = db['dataset']

            context = {
                "connection": str(client)
            }
            return render(request, "management/index.html", context)
        except:
            return render(request, "error/error404.html")
    else:
        return render(request, "management/index.html")


def disconnect(request):
    global client
    if client is not None:
        client = client.close()
    context = {
        "connection": str(client)
    }
    return redirect('/management/', context)


def get_client():
    if client is not None:
        return client
    else:
        return None
