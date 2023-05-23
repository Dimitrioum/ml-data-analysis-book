from django.urls import path
from . import views

app_name = 'bokeh_app'

urlpatterns = [
 path('', views.plot_page, name='plot_page')
]