import pickle
import os
import torch
from torch.backends import cudnn
from .config import DEVICE
from .models.license_classifier.cnn import Network


# декодировщик, словарь эталонов
pickles = ["CarPlateDictionary.pkl"]

# Имена загружаемых данных в словарь
pickle_names = ["ocr_decoder"]


def upload_data():
    """
    Загрузка необходимых данных для работы модели.
    :return: Возвращает словарь, содержащий в себе: ocr_decoder: CarPlateDictionary.pkl
    """

    files = dict()
    for pck, name in zip(pickles, pickle_names):
        with open(os.path.join("sidedata", pck), "rb") as f:
            d = {name: pickle.load(f)}
            files.update(d)
    return files


def load_cnn_model(weights, decoder):
    """
    Загрузка Pytorch модели.
    :param decoder: (dict) словарь для декодирования символов.
    :param weights: (string) наименование весов.
    :return: CNN модель, готовая к применению.
    """

    net = Network(num_classes=22, input_channels=1, device=DEVICE, decoder=decoder)
    net.load_state_dict(torch.load(os.path.join("sidedata", weights), map_location=DEVICE))
    print("Loading OCR pretrained network...")
    if DEVICE.type == 'cuda':
        cudnn.benchmark = True
        net = net.cuda()
    return net
