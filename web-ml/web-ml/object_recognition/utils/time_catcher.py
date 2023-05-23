
import cv2
import numpy as np
from imutils import contours
import PIL as pil

LOWER_CLR = np.array([0, 0, 0], dtype="uint8")
UPPER_CLR = np.array([40, 255, 40], dtype="uint8")# Зеленый


class TimestampCatcher(object):

    def __init__(self):

        pass

    def bitwise(self, frame, lower, upper):
        self.frame = frame
        self.lower = lower
        self.upper = upper

        # Маска выделения
        mask = cv2.inRange(frame, lower, upper)
        return mask

    def rgb2grayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), frame

    def get_contours(self, input_img):
        self.input_img = input_img
        cnts = cv2.findContours(input_img, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if cv2.__version__[0] == '4' else cnts[1]
        cnts = contours.sort_contours(cnts, method="left to right")[0]
        return cnts

    def resize(self, inut_img, x=50, y=50):
        self.inut_img = inut_img
        return np.asarray(pil.Image.fromarray(inut_img).resize((x, y),
                                                               pil.Image.NEAREST)).reshape(x*y,)

    def threshold(self, input_img, border=100):
        self.input_img = input_img
        _, threshold = cv2.threshold(input_img, border, 255, cv2.THRESH_BINARY)
        return threshold

    def get_segments(self, input_img, cnts, max_x=0, max_y=0):
        self.input_img = input_img
        self.cnts = cnts

        # input_img   изображение в градациях серого
        # output_img   изображения в rgb
        chars = []
        for cnt in cnts:
            # Поиск координаты левой верхней точки, ширины, правой нижней точки и высоты
            (x, y, w, h) = cv2.boundingRect(cnt)
            # Фильтрация найденных знаков
            if (h/w >= 1.005) and (h/w <= 2.5) and (h >= 15) and (w >= 1.0005):
                # Бинаризаци изображения
                # Передается маска, бинаризация уже в этом варианте не нужна
                threshold_img = input_img[y:y+h,x:x+w]
                # Добавить знак в список
                chars.append(self.resize(threshold_img))
        return chars

    def recognizetime(self, image, model):
        self.model = model

        """Классификатор цифр для записи времени с кадра"""
        """
        Args:
          image (numpy array): слайс кадра с локализацией записанного времени
          model (scikit learn obj): модель классификации

        Returns:
          preds (string): "сырая"" строка распознанных цифр
          proba (float): усредненная вероятность принадлежности к классам всех распознанных цифр
        """

        # Разделяем побитово кадр по заданному диапазону и выделяем интервал белого для поиска времени
        bitwise_img = self.bitwise(image, LOWER_CLR, UPPER_CLR)

        # Получаем координаты 4 точек каждого найденного объекта, отсортированные по оси x
        cnts = self.get_contours(bitwise_img)

        # Получаем список найденных значений номеров
        chars = self.get_segments(bitwise_img, cnts)

        """
        for i in range(len(cnts)):
            (x, y, w, h) = cv2.boundingRect(cnts[i])
            if (h/w >= 1.005) and (h/w <= 2.5) and (h >= 15) and (w >= 1.0005):
                cv2.rectangle(image, (x-1, y-1), (x+w, y+h), (0, 255, 0), 1)
        """

        # Получить значения знаков
        if len(chars) == 6:
            if np.mean(np.amax(model.predict_proba(chars), axis=1)) >= 0.5:
                preds = ''.join(model.predict(chars).astype(str))
                return preds, np.mean(np.amax(model.predict_proba(chars), axis=1))
            else:
                return '-', 0.
        else:
            return '-', 0.
