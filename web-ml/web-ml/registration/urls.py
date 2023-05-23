from django.urls import path
from . import views

app_name = 'registration'

urlpatterns = [
    path(r'', views.registration_page, name='registration_page'),
]
