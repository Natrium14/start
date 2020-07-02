from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index_management'),
    path('create_connection', views.create_connection, name='create_connection'),
]
