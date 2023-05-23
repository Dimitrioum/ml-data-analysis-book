from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic.edit import FormView
from object_recognition.views import recognition_page
from bokeh_app.views import plot_page


def start_page(request):
	return render(request, 'start_page_new.html', )