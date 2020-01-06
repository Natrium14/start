from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index_dataset'),
    path('model_test/', views.model_test, name='model_test')
]
