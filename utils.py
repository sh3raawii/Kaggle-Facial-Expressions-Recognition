# Author: Mostafa Mahmoud Ibrahim Hassan
# Email: mostafa_mahmoud@protonmail.com

import numpy as np
import cv2 as cv


def square_image(image):
    """
    return a square image by zero padding the smaller dimension
    :param image: numpy array of any size
    :return image: numpy array assuring width == height
    :return padding: tuple of (top, bottom, left, right) padding values
    """
    height, width = image.shape[0], image.shape[1]
    padding = (0, 0, 0, 0)
    if height > width:
        border_size = (height - width) / 2
        padding = (0, 0, np.int(np.floor(border_size)), np.int(np.ceil(border_size)))
        image = cv.copyMakeBorder(image, padding[0], padding[1], padding[2], padding[3], cv.BORDER_CONSTANT,
                                  value=(0, 0, 0))
    elif width > height:
        border_size = (width - height) / 2
        padding = (np.int(np.floor(border_size)), np.int(np.ceil(border_size)), 0, 0)
        image = cv.copyMakeBorder(image, padding[0], padding[1], padding[2], padding[3], cv.BORDER_CONSTANT,
                                  value=(0, 0, 0))
    return image, padding
