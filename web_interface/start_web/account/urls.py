from django.urls import path

from account import views

urlpatterns = [
    path('login', views.login_view, name='login'),
    path('logout', views.logout_view, name='logout'),
    path('register', views.register, name='register'),
    path('cabinet', views.cabinet, name='cabinet'),
    path('change_password', views.change_password, name='change_password')
]