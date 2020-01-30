from django.urls import path, re_path

from . import views

urlpatterns = [
    path('', views.index, name='index_dataset'),
    path('upload_data', views.upload_data, name='upload_data'),
    path('stat_index', views.stat_index, name='stat_index'),
    path('show_dataset', views.show_dataset, name='show_dataset'),
    path('model_test', views.model_test, name='model_test'),
    path('make_plot.png', views.make_plot, name='make_plot')
]
