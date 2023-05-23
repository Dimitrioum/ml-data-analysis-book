from os.path import join
from os import getcwd
from sys import path
import torch


ROOT_PREFIX = getcwd()
DATASET = join(ROOT_PREFIX, "dataset")
DBNAME = "sqlite:///utils/sqlite/dbsqlite.db"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    print("WARNING: CUDA ACTIVATED")
else:
    print("WARNING: CUDA NOT ACTIVATED")

NOMEROFF_NET_DIR = join(ROOT_PREFIX, 'nomeroff-net')
MASK_RCNN_DIR = join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_LOG_DIR = join(NOMEROFF_NET_DIR, 'logs')
path.append(NOMEROFF_NET_DIR)

# Кофигуратор ocr, работа с контурами объектов
# find_char_contours
# Граница отсева контуров регионов по ширине
CNT_W = 2
# Нижняя граница отсева конутров по высоте
LB_H = 5
# Верхняя граница отсева контуров по высоте
HB_H = 40

NBRS = 8
BGRND = 0

# Частота обновления в папку dataset
UPDATE_FREQ = 30*60*10  # 30м*60с*теорет.фпс
# Количество каждых N кадров, которые пропускаются алгоритмом распознавания
SKIP_FRAMES = 5

# Margin для детектора авто. Для расширения рамки авто.
REC_MARGIN = 50

# Расширение рамки захватывающей символ номера на INDENTION пикселей
INDENTION = 0

# Размах диагонали номера автомобиля рамки распознавания
MINDIST = 50
MAXDIST = 160

MNSSD = 'MN1SSD512-E30-R460-C1939.pth'

# Корректировка времени от UTC по геокодам АЗС
UTC_UPD = {
    "10": 3,
    "11": 3,
    "12": 3,
    "13": 3,
    "14": 3,
    "15": 3,
    "20": 5,
    "21": 5,
    "22": 5,
    "24": 5,
    "25": 5,
    "30": 7,
    "31": 6,
    "32": 7,
    "33": 7,
}

"""
-------------------------------
АЗС без знаков отличия колонок:
-------------------------------
10108:
Колонки 1,2 - камера 22, крайняя правая.
Колонки 3,4 - камера 21, посередине
Колонки 5,6 - камера 20, крайняя левая
"""
