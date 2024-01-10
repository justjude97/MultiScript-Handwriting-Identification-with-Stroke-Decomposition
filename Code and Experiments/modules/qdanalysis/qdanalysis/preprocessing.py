"""
preprocessing.py

holds all preprocessing code for the different datasets
"""

import cv2 as cv
import numpy as np

def preprocess(image: np.ndarray):
    #2d array assumed to be grayscale image
    if len(image.shape) < 3:
        #i don't think an explicit copy is needed
        image_copy = image
    else:
        print(len(image.shape))
        image_copy = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #threshold and invert image at the same time to work with foreground elements (value=1)
    return cv.threshold(image_copy, 0, 1, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] 