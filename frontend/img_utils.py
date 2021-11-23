import cv2
import numpy as np


def create_silhouette(filter_img):
    # convert to gray
    gray = cv2.cvtColor(filter_img, cv2.COLOR_RGB2GRAY)
    gray[:, 0:20] = 0
    _, width = gray.shape
    gray[:, width - 20: width] = 0
    gray[gray == 0] = 1
    cv2.floodFill(gray, None, (15, 15), 0)
    gray[gray != 0] = 255

    kernel = np.dstack((gray, gray))
    kernel = np.dstack((kernel, gray))
    return kernel


def and_img(img, filter_img):
    result = img * filter_img
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def load(img_path):
    return cv2.imread(img_path)


