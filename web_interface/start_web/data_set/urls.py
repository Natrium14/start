from django.urls import path, re_path

from data_set import views

urlpatterns = [
    path('', views.main_page, name='main_page'),
    path('index', views.index, name='index_dataset'),
    path('model_training', views.model_training, name='model_training'),
    path('upload_data', views.upload_data, name='upload_data'),
    path('stat_index', views.stat_index, name='stat_index'),
    path('show_dataset', views.show_dataset, name='show_dataset'),
    path('model_train', views.model_train, name='model_train'),
    path('model_test', views.model_test, name='model_test'),
    path('make_plot', views.make_plot, name='make_plot'),
    path('vis_dbscan', views.vis_dbscan, name='vis_dbscan'),
]
