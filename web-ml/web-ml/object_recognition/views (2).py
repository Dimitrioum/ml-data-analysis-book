from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.generic.edit import FormView
from django.core.files import File
from django.core.files.uploadedfile import InMemoryUploadedFile
from .forms import DocumentForm, LoginForm
from .models import Document, MediaFile, DocumentOutput
from datetime import datetime
import subprocess
from django.conf import settings
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required

from urllib.request import pathname2url
from django.http import HttpResponse

from vda_django.settings import LOGIN_URL, LOGIN_REDIRECT_URL

from yolov5.utils.datasets import *
from yolov5.utils.utils import *

from shutil import move, copyfile
from mimetypes import MimeTypes

import sys
from time import sleep

from tqdm import tqdm
import base64

import ast
from io import StringIO
import pathlib

from celery import shared_task
from celery_progress.backend import ProgressRecorder
from celery.result import AsyncResult
from celery_progress.backend import Progress
from django.views.decorators.cache import never_cache
import json

PATTERNS = ['Ведро', 'Бутыль', 'Канистра', 'Сливной шланг']
FPS_DEFAULT = 12

@never_cache
def get_progress(request, task_id):
    progress = Progress(AsyncResult(task_id))
    return HttpResponse(json.dumps(progress.get_info()), content_type='application/json')


def get_checked_values(request):
	if request.is_ajax():
		items = ast.literal_eval(list(request.POST.items())[0][0])
		return HttpResponse(str(selected_elements))


@shared_task(bind=True)
def recognize_objects(self, filenames: list, classes: list, device_type='0'):
	progress_recorder = ProgressRecorder(self)
	device_type = device_type
	# out = 'documents/output/' + datetime.now().strftime('%Y/%m/%d')
	out = 'tmp'
	source = ','.join([settings.PROJECT_ROOT + filename for filename in filenames])
	weights = settings.WEIGHTS_FOLDER + "best.pt"
	print(weights)
	imgsz = 640

	device = torch_utils.select_device(device_type)
	if os.path.exists(out):
		pass
	try:
		os.makedirs(out)  # make new output folder
	except PermissionError:
		os.makedirs(out)
	except FileExistsError:
		pass

	half = device.type != 'cpu'  # half precision only supported on CUDA
	model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
	model.to(device).eval()
	if half:
		model.half()  # to FP16

	vid_path, vid_writer = None, None
	save_img = True
	dataset = LoadImages(source, img_size=imgsz)

	names = model.names if hasattr(model, 'names') else model.modules.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

	t0 = time.time()
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
	_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
	__label = None
	__xyxy = None
	__colors = None
	__im0 = None
	for path, img, im0s, vid_cap in dataset:
		# if (dataset.frame % 2 == 0) or (dataset.frame == 1):
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		t1 = torch_utils.time_synchronized()
		pred = model(img, augment=True)[0]

		pred = non_max_suppression(pred, 0.6, 0.5,
                                   fast=True, classes='', agnostic=True)
		t2 = torch_utils.time_synchronized()

		for i, det in enumerate(pred):  # detections per image
			p, s, im0 = path, '', im0s
			save_path = str(Path(out) / Path(p).name)
			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
			if det is not None and len(det):
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += '%g %ss, ' % (n, names[int(c)])  # add to string

				for *xyxy, conf, cls in det:
					__colors = colors[int(cls)]
					__xyxy = xyxy
					__label = '%s %.2f' % (names[int(cls)], conf)
					# __im0 = im0
					if save_img or view_img:  # Add bbox to image
						if names[int(cls)] in classes:
							label = '%s %.2f' % (names[int(cls)], conf)
							plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

			print('%sDone. (%.3fs)' % (s, t2 - t1))

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
		# else:
		# 	plot_one_box(__xyxy, im0s, label=__label, color=__colors, line_thickness=3)
		# 	vid_writer.write(im0s)
		progress_recorder.set_progress(dataset.frame + 1, dataset.nframes)

	if save_img:
		print('Results saved to %s' % os.getcwd() + os.sep + out)

	print('Done. (%.3fs)' % (time.time() - t0))
	if vid_writer:
		vid_writer.release()
	# out = 'documents/output/' + datetime.now().strftime('%Y%m%d')
	# full_path = os.getcwd() + os.sep + out + '\\azs2.mp4'
	# return full_path
	# newdoc.save()
	outdoc = DocumentOutput()
	print(save_path.split('.')[0] + '.avi')
	with open(save_path.split('.')[0] + '.avi', 'rb') as out_file:
		outdoc.outputfile.save(save_path.split('\\')[-1], File(out_file), save=True)
	outdoc.save()
	return redirect('/object_recognition/')


def update_progress(self, proc):
	# Create progress recorder instance
	progress_recorder = ProgressRecorder(self)

	while True:
		# Read wget process output line-by-line
		line = proc.stdout.readline()

		# If line is empty: break loop (wget process completed)
		if line == b'':
			break

		linestr = line.decode('utf-8')
		print(linestr)
		if 'video' in linestr:
			# Find percentage value using regex
			progress = log.split(')')[0].split('(')[-1]
			# Print percentage value (celery worker output)
			print(progress)
			# Build description
			current_frame = int(progress.split('/')[0])
			total_frames  = int(progress.split('/')[1])
			percentage = round(current_frame * 100 / total_frames)

			progress_description = 'Распознавание (' + str(percentage) + '%)'
			# Update progress recorder
			progress_recorder.set_progress(int(percentage), 100, description=progress_description)
		else:
			# Print line
			print(linestr)

		# Sleep for 100ms
		time.sleep(0.1)


@shared_task(bind=True)
def process_recognition_task(self, source : str, classes : list, output_dir : str, device_type="cpu"):
	print('Recognition task started')

	recognition = subprocess.Popen(["python",
									"yolov5/detect.py",
									"--weights",
									# "yolov5/weights/best_azs_skytrack.pt",
									"yolov5/weights/best.pt",
									"--img",
									"640",
									"--conf",
									"0.6",
									"--source",
									source,
									"--device",
									device_type,
									"--output",
									output_dir,
									"--classes",
									classes,
									], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

	update_progress(self, recognition)

	recognition.terminate()
	try:
		# Wait 100ms
		recognition.wait(timeout=0.1)
		# Print return code (celery worker output)
		print(f'Subprocess terminated [Code {recognition.returncode}]')
	except subprocess.TimeoutExpired:
		# Process was not terminated in the timeout period
		print('Subprocess did not terminate on time')

	# Check if process was successfully completed (return code = 0)
	if recognition.returncode == 0:
		# Return message to update task result
		return 'Процесс распознавания прошел успешно!'
	else:
		# Raise exception to indicate something wrong with task
		raise Exception('Recognition timed out, try again')


# @login_required(login_url=LOGIN_URL, redirect_field_name='recognition_page')
def recognition_page(request):
	patterns = ['Ведро', 'Бутыль', 'Канистра', 'Сливной шланг']
	sys.path.insert(0, str(pathlib.Path().absolute()) + "/yolov5")
	message = None
	if request.is_ajax():
		try:
			keys = ast.literal_eval(list(request.POST.keys())[0])
			print(keys)
		except IndexError:
			return redirect('/object_recognition/')
		if isinstance(keys[0], list):
			# recognize_task = recognize_objects.delay(filenames=keys[0], classes=keys[1], device_type='cpu')
			# out = 'tmp'
			# recognition_task = process_recognition_task.delay(source=keys[0], classes=keys[1], output_dir=out)
			# task_id = recognition_task.task_id
			# print('task_id')
			# print(task_id)
			# return render(request, 'list.html', locals())
			recognize_objects(filenames=keys[0], classes=keys[1], device_type='cpu')

	elif request.method == 'POST':
		form = DocumentForm(request.POST, request.FILES)
		files = request.FILES.getlist('docfile')
		if form.is_valid():
			for file in files:
				newdoc = Document(docfile=file)
				newdoc.save()

            # Redirect to the document list after POST
			return redirect('/object_recognition/')
			message = 'Файлы загружены'
	else:
		form = DocumentForm()

	# Document.objects.all().delete()
	documents = Document.objects.all()
	documents_output = DocumentOutput.objects.all()
	return render(request, 'list.html', locals())


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
