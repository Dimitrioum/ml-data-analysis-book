from django.urls import path
from . import views

app_name = 'object_recognition'

urlpatterns = [
    path(r'drain_control/', views.recognition_page, name='recognition_page'),
    # path(r'', views.get_checked_values, name='get_checked_values'),
    path(r'car_plates/', views.car_plates_page, name='car_plates_page'),
    path(r'login/', views.user_login, name='user_login'),
]
