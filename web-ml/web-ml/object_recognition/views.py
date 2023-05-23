from django.http import HttpResponse
from django.template import loader
from django.views.decorators.cache import never_cache
from django.conf import settings
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.generic.edit import FormView
from django.core.files import File
from django.core.files.uploadedfile import InMemoryUploadedFile
from .forms import DocumentForm, LoginForm, DocumentFormCarPlates
from .models import Document, MediaFile, DocumentOutput, CarPlatesRecInput, CarPlatesRecOutput

from datetime import datetime
from contextlib import redirect_stdout
from urllib.request import pathname2url
from vda_django.settings import LOGIN_URL, LOGIN_REDIRECT_URL, PATTERNS, FPS_DEFAULT
from shutil import move, copyfile
from mimetypes import MimeTypes
from io import StringIO

from yolov5.utils.datasets import *
from yolov5.utils.utils import *

from .class_log import Log
from .utils.config import SKIP_FRAMES
from .utils import sidedata  # Модуль загрузки данных
from .utils import plate_detector_class  # Скрипты распознавания
from .utils.ocr_container_class import ContainerOCR

import sys
import subprocess
import base64
import ast
import pathlib
import json
import os
import time

import asyncio

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_checked_values(request):
	if request.is_ajax():
		items = ast.literal_eval(list(request.POST.items())[0][0])
		print('items from get_checked_values')
		print(items)
		return HttpResponse(str(selected_elements))


# @shared_task(bind=True)
def recognize_objects(filenames: list, classes: list, device_type='0'):
	""" Функция для распознавания объектов на видео

	Args:
	    filenames: список, содержащий пути к выбранным файлам
	    classes: список, содержащий лейблы объектов, которые будут отрисовываться на проанилизированно видео
	    device: тип устройства, на котором будут запущены вычисления - 'cpu' для запуска на ЦПУ; '0', '1' и т.д. для выбора определенного ГПУ
	"""
	device_type = device_type
	source = ','.join([settings.PROJECT_ROOT + filename for filename in filenames])
	# путь к расположению весов обученной нейронной сети
	weights = settings.WEIGHTS_FOLDER + "best.pt"
	# высота изображения, до которого будет уменьшено/увеличено оригинальное изображение из видео
	imgsz = 640
	device = torch_utils.select_device(device_type)
	# используем half-точность, если выбрано ГПУ для вычислений
	half = device.type != 'cpu'
	# загружаем модель (по умолчанию в формате FP32)
	model = torch.load(weights, map_location=device)['model'].float()
	model.to(device).eval()
	if half:
		model.half()  # to FP16

	vid_path, vid_writer = None, None
	save_img = True
	# инициализируем датасет, генерирующий изображения из видео с измененным разрешением
	dataset = LoadImages(source, img_size=imgsz)
	# лейблы в обучающем датасете
	names = model.names if hasattr(model, 'names') else model.modules.names
	# список с кодировками цветов для отрисовки распознанных объектов
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
	# отмечаем момент времени начала процесса распознавания
	t0 = time.time()
	# инициализируем тестовое изображение
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)
	# прогоняем тестовое изображение через нейронную сеть
	_ = model(img.half() if half else img) if device.type != 'cpu' else None
	# начинаем проход по датасету
	for path, img, im0s, vid_cap in dataset:
		img = torch.from_numpy(img).to(device)
	    # меняем тип данных для хранения значений пикселей с uint8 на fp16/32
		img = img.half() if half else img.float()
	    # нормализация изображения : значения 0 - 255 в 0.0 - 1.0
		img /= 255.0
	    # увеличиваем размерность тензора изображения с 3 до 4
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

	    # отмечаем время, необходимое на обработку одного кадра видео
		t1 = torch_utils.time_synchronized()
		pred = model(img, augment=True)[0]

		pred = non_max_suppression(pred, 0.6, 0.5,
	                               fast=True, classes='', agnostic=True)
		t2 = torch_utils.time_synchronized()

	    # начинаем проход по распознанным объектам на кадре видео
		for i, det in enumerate(pred):
			p, s, im0 = path, '', im0s
			save_path = str(Path(p).name)
			s += '%gx%g ' % img.shape[2:]
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
			if det is not None and len(det):
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Оформляем результаты распознавания для логов
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += '%g %ss, ' % (n, names[int(c)])  # add to string

	            # Отрисовываем рамки вокруг распознанных объектов
				for *xyxy, conf, cls in det:
					if save_img or view_img:
						# TODO доделать фильтр распознаваемых объектов
						# if names[int(cls)] in classes:
						label = '%s %.2f' % (names[int(cls)], conf)
						plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

			print('%sDone. (%.3fs)' % (s, t2 - t1))

	        # сохраняем видео с отрисованными рамками вокруг распознанных объектов, сохраняя оригинальную частоту кадров
			if save_img:
				if dataset.mode == 'images':
					cv2.imwrite(save_path, im0)
				else:
					if vid_path != save_path:  # new video
						vid_path = save_path
						if isinstance(vid_writer, cv2.VideoWriter):
							vid_writer.release()  # release previous video writer

						fps = vid_cap.get(cv2.CAP_PROP_FPS)
						w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
						h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
						save_path = save_path.split('.')[0] + '.avi'
						vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) # *'mp4v'
					vid_writer.write(im0)

	if save_img:
		print('Results saved to %s' % os.getcwd() + os.sep + out)

	print('Done. (%.3fs)' % (time.time() - t0))
	if vid_writer:
		vid_writer.release()
	# сохраняем видео в модель
	outdoc = DocumentOutput()
	with open(save_path.split('.')[0] + '.avi', 'rb') as out_file:
		outdoc.outputfile.save(save_path.split('\\')[-1], File(out_file), save=True)
	outdoc.save()
	return redirect('/object_recognition/')


def get_object_timeline_in_seconds(numbered_events: dict, fps: int = FPS_DEFAULT) -> dict:
    """
    Args:
        numbered_events: словарь, в ключах закодированы имена распознаванных объектов и порядковый номер события с таким объектом
                         (например, 'bottle_1'), в значениях списки с номерами кадров, соответствующие появлению объекта на видео
                         и момента, когда объект пропал с видео (например, [10,134])
        fps: частота кадров в анализируемом видео

    Returns:
        events_seconds: словарь, в ключах закодированы имена распознаванных объектов и порядковый номер события с таким объектом
                        (например, 'bottle_1'), в значениях списки с моментом времени в секундах, когда объект появвился на видео
                        и момент времени в секундах, когда он пропал с видео
    """
    events_seconds = {}
    for key in numbered_events.keys():
        frames_start_end = numbered_events[key]
        # делим номер кадра на частоту кадров
        events_seconds[key] = [int(frames_start_end[i] / fps) for i in [0, 1]]
    return events_seconds


def get_object_frames_numbers(events: dict, fps: int = FPS_DEFAULT) -> dict:
	"""
	Args:
	    events: словарь, в ключах закодированы имена распознаванных объектов и порядковый номер события с таким объектом
	                     (например, 'bottle_1'), в значениях списки с номерами кадров, соответствующие появлению объекта на очередном кадре
	                     из видео (например, [10,11,12,13,14,134,135,136,137])
	    fps: частота кадров в анализируемом видео

	Returns:
	    numbered_events: словарь, в ключах закодированы имена распознаванных объектов и порядковый номер события с таким объектом
	                     (например, 'bottle_1'), в значениях списки с номерами кадров, соответствующие появлению объекта на видео
	                     и момента, когда объект пропал с видео (например, [10,14])
	"""
	event_ids = {obj_name: event_id for obj_name, event_id in zip(events.keys(), [1] * len(events.keys()))}
	numbered_events = {obj_name + '_' + str(event_ids[obj_name]) : [events[obj_name][0]] for obj_name in events.keys()}
	for obj_name in events.keys():
		frames = events[obj_name]
		for i in range(1, len(frames)):
			# если объект пропадал из кадра меньше, чем на 2 секунды, делаем допущение, что объект находился все это время на сцене
			if ((frames[i] - frames[i - 1]) < (2 * fps)):
				if (i + 1) == len(frames):
					if obj_name + '_' + str(event_ids[obj_name]) in numbered_events.keys():
						numbered_events[obj_name + '_' + str(event_ids[obj_name])].append(frames[i])
						event_ids[obj_name] += 1
					else:
						numbered_events[obj_name + '_' + str(event_ids[obj_name])] = [frames[i]]
				continue
			if (obj_name + '_' + str(event_ids[obj_name])) in numbered_events.keys():
				numbered_events[obj_name + '_' + str(event_ids[obj_name])].append(frames[i - 1])
				event_ids[obj_name] += 1
				numbered_events[obj_name + '_' + str(event_ids[obj_name])] = [frames[i]]
			else:
				numbered_events[obj_name + '_' + str(event_ids[obj_name])] = [frames[i]]
	return numbered_events


def get_object_timeline(raw_logs_path: str) -> dict:
    """
    Args:
        raw_logs_path: путь к файлу, содержащему логи процесса распознавания объектов на видео

    Returns:
        events: словарь, в ключах закодированы имена распознаванных объектов и порядковый номер события с таким объектом
                         (например, 'bottle_1'), в значениях списки с номерами кадров, соответствующие появлению объекта на очередном кадре
                         из видео (например, [10,11,12,13,14,134,135,136,137])
    """
    with open(raw_logs_path, 'r') as raw_logs:
        lines = raw_logs.readlines()
        count = 0
        events = {}
        event_id = 0
        for line in lines:
            if line.startswith('Results'):
                break
            elif line.startswith('video'):
                progress = line.split(')')[0].split('(')[-1]
                current_frame = int(progress.split('/')[0])
				# 8 is the number of characters in the description of the frame dimension
                recognized_objects = line.split(':')[-1].split('Done')[0][8:].split(',')
                if recognized_objects[0] == ' ':
                    continue
                else:
                    for obj in recognized_objects:
                        items = [item for item in obj.split(' ') if item]
                        if items:
                            obj_name = items[-1]
                            if obj_name in events.keys():
                                events[obj_name].append(current_frame)
                            else:
                                events[obj_name] = [current_frame]
    return events


# @login_required(login_url=LOGIN_URL, redirect_field_name='recognition_page')
def recognition_page(request):
	# делаем директорию /yolov5 видимым модулем
	sys.path.insert(0, str(pathlib.Path().absolute()) + "/yolov5")
	message = None
	if request.is_ajax():
		try:
			keys = ast.literal_eval(list(request.POST.keys())[0])
			print(keys)
		except IndexError:
			return redirect('/object_recognition/drain_control/')
		if isinstance(keys[0], list):
            # перенаправляем логи из командной строки в файл
			with open('raw_logs.txt', 'w') as f:
				with redirect_stdout(f):
					# TODO create async func
					# how to handle logs in async mode ?
					recognize_objects(filenames=keys[0], classes=keys[1], device_type='cpu')
			timeline_in_seconds = get_object_timeline_in_seconds(get_object_frames_numbers(get_object_timeline(raw_logs_path='raw_logs.txt')))
			data = {'response': timeline_in_seconds}
			return JsonResponse(data)

	elif request.method == 'POST':
		form = DocumentForm(request.POST, request.FILES)
		files = request.FILES.getlist('docfile')
		if form.is_valid():
			for file in files:
				newdoc = Document(docfile=file)
				newdoc.save()

			# перенаправляем пользователя на начальную страницу после загрузки файлов
			return redirect('/object_recognition/drain_control/')
			message = 'Файлы загружены'
	else:
		form = DocumentForm()

	documents_drain_control = Document.objects.all()
	documents_output_drain_control = DocumentOutput.objects.all()

	return render(request, 'drain_control.html', locals())


def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            user = authenticate(username=cd['username'], password=cd['password'])
            if user is not None:
                if user.is_active:
                    login(request, user)
                    return redirect('/object_recognition/')
                else:
                    return HttpResponse('Disabled account')
            else:
                return HttpResponse('Invalid login')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})


async def recognize_car_plates(filenames: list, device_type: str, debug: bool = False):
	""" Функция для распознавания номеров автомобилей на видеозаписи
	Args:
		filenames: список, содержащий пути (типа string) к выбранным для распознавания видеозаписям
		device_type: строка, обозначающая устройство, на котором будут выполнены вычисления - 'cpu' для использования ЦПУ
		 			 и '0', '1' для выбора конкретного графического процессора
	"""
	cwd = os.getcwd()
	print(cwd)
	# Путь открытия детектора
	ldp = [os.path.join("object_recognition", "utils", "models", "license_bbox", x) for x in os.listdir(os.path.join("object_recognition", "utils", "models", "license_bbox"))]
	ldp = dict((k, v) for k, v in enumerate(ldp))
	license_detector_paths = {str(v).split(".")[1]: v for k, v in ldp.items()}
	# Путь открытия детектора
	cascade_path = os.path.join("object_recognition", "utils", "models", "license_cascade", "haarcascade_russian_plate_number.xml")
	# Путь до логов
	# save_path = os.path.join(cwd, 'logs', f'azs_{azs_number}_trk_{trk_number}_log.csv')
	save_path = os.path.join("object_recognition", "logs", f'car_plates_recognition_logs.csv')

	with open(save_path, 'a', encoding='utf-8', newline='', buffering=1) as log_file:
		log = Log(log_file)
		# Запуск алгоритма по открытому каналу ip-камеры

		# Загрузка моделей и шаблонов распознавания
		# files = sidedata.upload_data()

		# Декодер классов знаков номера авто
		# ocr_decoder = ContainerOCR()  # files.get(list(files.keys())[0])

		# Подгрузка модели
		# cnn_grayscale = sidedata.load_cnn_model("GrayscaleConvNet.pth", ocr_decoder)

		# Инициализация детектора
		detector = plate_detector_class.PlateDetector(
            save_result=True,
            plate_confidence_threshold=0.5,
            class_confidence_threshold=0.9,
            nms_thresh=0.45
        )

		# Список видеозаписей в папке
		# TODO os.path.abspath doesn't work on Mac
		# video_list = [os.path.abspath(filename) for filename in filenames]
		video_list = ["/Users/user/Documents/safedelivery-vda" + filename for filename in filenames]
		print("Total videos: ", len(video_list))
		for i, video_name in enumerate(video_list):
			print(f"Current video {i + 1}/{len(video_list)}\n{video_name}")
			video_path = video_list[i]
			# detector.capture_plate(video_path, cascade_path, ocr_decoder, SKIP_FRAMES, debug, log)
			plate_recognition_dict = detector.infer(video_path)
			# print(plate_recognition_dict)


def car_plates_page(request):
	if request.is_ajax():
		videos_paths_to_process = ast.literal_eval(list(request.POST.keys())[0])
		print(videos_paths_to_process)
		asyncio.run(recognize_car_plates(filenames=videos_paths_to_process, device_type='0'))

	elif request.method == 'POST':
		form = DocumentFormCarPlates(request.POST, request.FILES)
		files = request.FILES.getlist('car_plates_file_input')
		print('files from form')
		print(files)
		if form.is_valid():
			for file in files:
				newdoc_car_plate = CarPlatesRecInput(car_plates_file_input=file)
				newdoc_car_plate.save()

			return redirect('/object_recognition/car_plates/')
			message = 'Файлы загружены'
	else:
		form = DocumentFormCarPlates()

	documents_car_plates = CarPlatesRecInput.objects.all()
	print(documents_car_plates)
	documents_output_car_plates = CarPlatesRecOutput.objects.all()
	print(documents_output_car_plates)


	return render(request, 'car_plates.html', locals())
