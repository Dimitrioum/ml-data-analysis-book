import cv2
from numpy import isin, zeros
from skimage.filters import threshold_local


def clahe_image_filter(image):
    """Применение фильтрации с помощью CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Adaptive histogram equalization (AHE) improves on this by transforming each pixel
    with a transformation function derived from a neighbourhood region.
    It was first developed for use in aircraft cockpit displays
    Args:
      image (numpy array): первичный numpy массив изображения

    Returns:
      image (numpy array): бинаризованный массив изображения

    """
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Перевод в формат цветового пространства CIE LAB
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_thresh(image):
    """
    Args:
      image (numpy array): первичный numpy массив изображения

    Returns:
      image (numpy array): бинаризованный массив изображения

    """
    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method="gaussian")
    th = (V > T).astype("uint8") * 255
    return cv2.bitwise_not(th)


def get_adaptive_thresh(image):
    """
    Args:
      image (numpy array): первичный numpy массив изображения
    Returns:
      image (numpy array): бинаризованный массив изображения
    """

    return cv2.adaptiveThreshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )


def get_cv2_ordinary_thresh(image):
    """
    Args:
      image (numpy array): первичный numpy массив изображения
    Returns:
      image (numpy array): бинаризованный массив изображения
    """
    _, th = cv2.threshold(
        cv2.cvtColor(clahe_image_filter(image), cv2.COLOR_BGR2GRAY),
        80,
        255,
        cv2.THRESH_BINARY_INV
    )
    return th


def composite_filtering(image):
    """Композитная бинаризация изоражения
    Args:
      image (numpy array): первичный numpy массив изображения
    Returns:
      image (numpy array): бинаризованный массив изображения
    """
    ord_thresh = get_cv2_ordinary_thresh(image)
    adaptive_thresh = get_adaptive_thresh(image)
    thresh = get_thresh(image)

    filt = isin(ord_thresh & adaptive_thresh | thresh & adaptive_thresh | thresh & ord_thresh, 255)
    filtered = zeros(adaptive_thresh.shape, dtype="uint8")
    filtered[filt] = 255
    return filtered
