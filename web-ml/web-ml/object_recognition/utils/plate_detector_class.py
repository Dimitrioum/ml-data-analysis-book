import cv2
import numpy as np
import datetime
from imutils import contours
from skimage import measure
from .models.license_cascade import CascadeBoundBox
from .models.ssd_mobilenet512.model_class import MobileNetSSD512
from .models.ocr import utils

from .models.ocr.options_detector import OptionsClassifier
from .models.ocr.text_detector import LicenceDetector
from .models.ocr.plate_detector import ObjectDetector
from .models.ocr.rect_detector import RectDetector

from .base64.base64_util import base64_encode
from .config import *
from .image_preprocess import composite_filtering
'''from utils.sqlite.sqlite_upd import SQLiteUpdater
from correct import correct_raw_data'''

from .models.ocr import inference_config as infer_config

import math
import skimage
import numpy as np
import tensorflow as tf

from .models.ocr.utils import parallel_compute, timeit, tf_limit_gpu_memory
from .models.ocr.config import MEM_LIMIT
from .models.ocr.mrcnn_v2 import utils as mrcnn_v2_utils
from .models.ocr.utils import timeit
# from ocr.mrcnn_v2 import model as model_mrcnn

tf_limit_gpu_memory(tf, MEM_LIMIT)

import imutils
import asyncio




class PlateDetector(object):
    """Детектор номеров автомобилей"""

    def __init__(self, save_result=False, plate_confidence_threshold=0.5, class_confidence_threshold=0.9, nms_thresh=0.45):
        self.plate_confidence_threshold = plate_confidence_threshold
        self.class_confidence_threshold = class_confidence_threshold
        self.nms_thresh = nms_thresh
        self.save_result = save_result  # True/False - включение сохранение прогноза по кадрам видеострима

        self.kwargs = infer_config.compile_inference_config()
        self.licence_region = 'ru'

        self.plate_detector = ObjectDetector(self.kwargs['object_detector'])
        self.rect_detector = RectDetector()
        self.options_model = OptionsClassifier(self.kwargs['options_model'], self.kwargs['weights']['options'])
        self.licence_model = LicenceDetector(self.kwargs['licence_model']['model'],
                                             self.kwargs['weights']['ocr'][self.licence_region])

    def denoise(self, image):
        """Однократная фильтрация изображения от шумов
        Args:
          image (numpy array): первичный numpy массив изображения
        Returns:
           image (numpy array): отфильтрованный numpy массив изображения
        """
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 2, 21)

    def read_plate(self, image, model_ocr):
        """Чтение детектированной области, потенцального номера автомобиля
        Args:
          image (numpy array): первичный numpy массив изображения
          model_ocr (obj): используемая модель классификации
        Returns:
           plate (string): строка распознанных знаков на кадре
           proba (float): средняя вероятность принадлежности символов к предсказанным класам
        """
        # Первичная обработка изображения
        # Композитная фильтрация GrayScale изображения
        preprocessed_image = composite_filtering(image)
        # Выделение регионов со связанными контурами
        charCnts = self.find_char_contours(preprocessed_image, CNT_W, LB_H, HB_H)
        if not charCnts:
            return None, None
        # Фильтрация выделенных контуров#
        # Параметры w_, h_l, h_h можно подтюнить
        plate, proba = self.find_char_cands(model_ocr, charCnts, preprocessed_image, INDENTION)
        if proba >= self.plate_confidence_threshold:
            return plate, proba
        return None, None

    def find_char_contours(self, thresh, w_, h_l, h_h):
        """Поиск контуров на binary изображении
        Args:
          thresh (int): изображение в binary
          w_ (int): граница отсева контуров регионов по ширине
          h_l (int): нижняя граница отсева конутров по высоте
          h_h (int): верхняя граница отсева контуров по высоте
        Returns:
          charCnts(list): список всех отобранных контуров с изображения
        """
        charCnts = []
        # labels = measure.label(thresh, neighbors=NBRS, background=BGRND)
        labels = measure.label(thresh, connectivity=2, background=BGRND)
        for label in np.unique(labels):
            # Если это связанный регион фона, пропустить его
            if label == 0:
                continue
            # Пропускаем лейблы, если связанное число элементов менее 10 и более 1000
            if sum(sum(labels == label)) < 10:
                continue
            if sum(sum(labels == label)) > 1000:
                continue

            # Создать маску для региона, чтобы отобразить только связанный регион на картинке
            # Создать контуры маски
            labelMask = np.zeros(thresh.shape, dtype="uint8")  # Создать матрицу изображения размером с binary thresh
            labelMask[labels == label] = 255  # Указанные координаты связанного региона отметить пикселями
            # Сохранить контуры каждого знака, оставив только 4 точки(CHAIN_APPROX_SIMPLE)
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            x, y, w, h = cv2.boundingRect(cnts[0])
            if w >= w_ and (h_l <= h <= h_h):  # Параметры фильтрации контуров
                charCnts.append(cnts[0])
        return contours.sort_contours(charCnts, method="left-to-right")[0] if 8 <= len(charCnts) <= 9 else None

    def find_char_cands(self, model_ocr, charCnts, thresh, offset):
        """
        Оценка списка контуров на кандидатов в строковые символы
        :param model_ocr: (obj) используемая модель распознавания
        :param charCnts: (list) список всех отобранных контуров с изображения
        :param thresh: (numpy array) массив numpy array бинарного изображения
        :param offset: (int) расширение рамки символа на offset пикселей
        :return:
            plate (string): строка предсказанных символов
            proba (float): средняя вероятность принадлежности символов к предсказанным класам
        """
        # Определить возможные символы и сделать оболочку для контура символов
        rois = []
        height, width = thresh.shape
        for (x, y, w, h) in [cv2.boundingRect(arr) for arr in charCnts]:
            x1 = clamp(x - offset, 0, width)
            x2 = clamp(x + w + offset, 0, width)
            y1 = clamp(y - offset, 0, height)
            y2 = clamp(y + h + offset, 0, height)
            rois.append(thresh[y1:y2, x1:x2])
        # return model_ocr.forward(model_ocr.transform(rois))
        return model_ocr.forward(rois)

    def infer(self, video_path: str) -> dict:
        """ Recognition of characters on license plate
        Args:
            video_path: path to the video file for processing
        Return:
            output_dict: dict with recognized characters and corresponding probabilities
        """
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, input_img = cap.read()
            detection_output = self.plate_detector([input_img])[0]
            output_dict = {'b64': None}
            if np.array(detection_output['masks']).shape[2] == 1:
                mask = utils.get_img_mask(detection_output)
                points_array = self.rect_detector.detect([mask])
                plate_roi = self.rect_detector.get_cv_zonesBGR(input_img, points_array)
                region, state, line = self.options_model(plate_roi)
                output_dict.update({'b64': utils.image2base64(plate_roi)})

                if region == self.licence_region:
                    licence_txt = self.licence_model(plate_roi)[0]
                    output_dict.update({'licence': licence_txt, 'region': region, 'state': state, 'line': line})
                    print(output_dict)
                else:
                    output_dict.update({'licence': 1, 'region': region, 'state': state, 'line': line})
                    print(output_dict)
            else:
                output_dict.update({'licence': 0, 'region': None, 'state': None, 'line': None})
                print(output_dict)





    def capture_plate(self, video_path, cascade_detector, model_ocr, skip_frames, debug, log):
        """Захват кадров видеострима для обработки
        Args:
        video_path (string): путь до видеофрагмента
        cascade_detector
        model_ocr (obj): используемая модель OCR
        azs_to_frame (int): номер азс
        trk_to_frame (int): номера ТРК
        skip_frames (int): количество пропускаемых нераспознаваемых кадров
        Returns:
        В случае параметра save_result=True при инициализации детектора происходит сохранение данных с кадров в .csv
        """
        # Объект захвата камеры
        cap = cv2.VideoCapture(video_path)
        win_name = video_path.split('/')[-1]
        if debug:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # Текущее разрешение камеры
        res_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        center_x = res_size[1] // 2

        # Детектор типов автомобилей на базе pytorch SSD-MobileNet
        car_bbox = MobileNetSSD512(MNSSD, res_size, score_thresh=self.class_confidence_threshold, nms_thresh=self.nms_thresh)
        # Детектор рамок автомобильного номера на базе каскадов Хаара
        haar_bbox = CascadeBoundBox(path=cascade_detector, mindist=MINDIST, maxdist=MAXDIST)

        '''# SQLiteUpdater БД
        sql_updater = SQLiteUpdater(DBNAME, DATASET)
        init_timestamp = datetime.datetime.utcnow()'''

        frame_num = 0

        while cap.isOpened():
            # Чтение раскадровки
            ret, frame = cap.read()

            if frame is None:
                break
            frame_num += 1

            if frame_num % skip_frames != 0:
                continue

            _, res = car_bbox(frame)

            # timestamp = datetime.datetime.utcnow() + datetime.timedelta(hours=UTC_UPD[str(azs_to_frame)[:2]])
            timestamp = datetime.datetime.utcnow()

            for class_name, class_confidence, (x1, y1, x2, y2) in res:
                # l = left, r = right
                # direction = trk_to_frame // 10 if x1 + ((x2 - x1) / 2) < center_x else trk_to_frame % 10
                plate, ocr_confidence, crop_img = None, None, None
                x_n, y_n, w_n, h_n = haar_bbox.cascade_bbox(frame[y1:y2, x1:x2])
                if x_n is not None:
                    crop_img = self.denoise(frame[y1 + y_n:y1 + y_n + h_n, x1 + x_n:x1 + x_n + w_n])
                    plate = model_ocr.cast(crop_img)
                    # TODO cast, confidence
                    plate, ocr_confidence = self.read_plate(crop_img, model_ocr)

                if self.save_result and plate:
                    log.add_record(
                        str(timestamp)[:19],
                        class_name,
                        round(class_confidence, 2),
                        plate,
                        round(ocr_confidence, 2),
                        base64_encode(crop_img),
                    )

                if debug:
                    # Обводка машины
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    # Класс машины
                    cv2.putText(
                        frame,
                        f"{class_name} {class_confidence * 100:.2f}%",
                        (x1, y1),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.5,
                        (255, 255, 255),
                        2
                    )

                    # Прорисовка области распознанного номера
                    if x_n is not None:
                        cv2.rectangle(frame, (x1 + x_n, y1 + y_n), (x1 + x_n + w_n, y1 + y_n + h_n), (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        f"{plate} {(ocr_confidence if ocr_confidence else 0) * 100:.2f}%",
                        (x1 + REC_MARGIN, y1 - REC_MARGIN),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.85,
                        (255, 50, 0),
                        2
                    )

            if debug:
                frame[:70, :400] = np.ones(frame[:70, :400].shape, dtype="uint8")
                cv2.putText(
                    frame,
                    f"{win_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.85,
                    (255, 255, 255),
                    2
                )
                cv2.putText(frame, f"{str(timestamp)[:19]}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
                cv2.imshow(f"{win_name}", frame)

            # TODO написать логику заполнения .csv

            # TODO
            """# Обновление БД
            if round((timestamp - init_timestamp).total_seconds()) % UPDATE_FREQ == 0:
                correct_raw_data(azs_to_frame, trk_to_frame)
                sql_updater.run()"""

            # Нажать q для выхода
            if cv2.waitKey(1) & 0xFF == ord("q"):
                if debug:
                    cv2.destroyAllWindows()
                break
