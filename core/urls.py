# urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('another_page/', views.another_page, name='another_page'),
     path('home/', views.another_page, name='home')
    # Add other URL patterns as needed
]
