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
    path('model_testing', views.model_test_page, name='model_testing'),
    path('model_selection', views.model_selection, name='model_selection'),
    path('make_plot', views.make_plot, name='make_plot'),
    path('make_plotly', views.make_plotly, name='make_plotly'),
    path('vis_model', views.vis_model, name='vis_model'),
    path('get_anomalies', views.get_anomalies, name="get_anomalies"),
    path('upload_data_db', views.upload_data_db, name="upload_data_db"),
    path('save_model', views.save_model, name="save_model"),
    path('normal_values', views.normal_values, name="normal_values"),
    path('vis_normal_values', views.vis_normal_values, name="vis_normal_values"),
    path('vis_model_plotly', views.vis_model_plotly, name="vis_model_plotly"),
]
