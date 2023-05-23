from django.db import models

class Document(models.Model):
    docfile = models.FileField(upload_to='documents/drain_control/input/%Y/%m/%d')

class MediaFile(models.Model):
	mediafile = models.CharField(max_length=1000)

class DocumentOutput(models.Model):
	outputfile = models.FileField(upload_to='documents/drain_control/output/%Y/%m/%d')

class VideoOutput(models.Model):
    video = models.URLField()

class CarPlatesRecInput(models.Model):
    car_plates_file_input = models.FileField(upload_to='documents/car_plates_rec/input/%Y/%m/%d')

class CarPlatesRecOutput(models.Model):
    car_plates_file_output = models.FileField(upload_to='documents/car_plates_rec/output/%Y/%m/%d')
