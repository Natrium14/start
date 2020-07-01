from django.urls import path, re_path

from description import views

urlpatterns = [
    path('contact', views.contact, name='contact'),
    path('about', views.about, name='about')
]