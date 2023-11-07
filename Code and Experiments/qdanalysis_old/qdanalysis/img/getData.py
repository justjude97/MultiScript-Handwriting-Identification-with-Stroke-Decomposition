import numpy as np
import pandas as pd
import cv2 as cv
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from qdanalysis.img.annotations import getImgList
from os.path import relpath

from skimage import img_as_float
from skimage.measure import shannon_entropy as entropy, find_contours, label
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_fill_holes

def getImages(srcDir: str):
    """
        gets all images within a directory and every subdirectory within it

        parameters:
        * srcDir - file path of the parent directory

        returns:
        * mappings - a dict mapping from the image path name to the index of the image
        * images - a list of all images in the source directory
    """
    
    imgList = getImgList(srcDir)

    images = []
    mapping = {}
    for idx, imgPath in enumerate(imgList):

        relImgPath = relpath(imgPath, srcDir)
        mapping[relImgPath] = idx

        image = load_img(imgPath, color_mode = "grayscale")

        image = img_to_array(image).astype(np.uint8)
        images.append(image)

    return mapping, images

def getDescriptors(images):

    #needs to be in a list. for now
    if type(images) != list:
        images = [images]

    #list containing all the lists of descriptors
    desList = []

    sift = cv.SIFT_create()
    for img in images:
        _, des = sift.detectAndCompute(img, None)
        
        if des is None:
            des = np.ndarray(shape=(0, 128))

        desList.append(des)

    return desList


def getWhiteBoxFeatures(images):
    """
        gets the 11 white box (macro features) from the individuality of handwriting paper
    """
    features = np.empty(shape=(len(images), 10))
    
    for idx, image in enumerate(images):
        image = img_as_float(image)
        
        features[idx, 0] = entropy(image)
        features[idx, 1] = threshold = threshold_otsu(image)

        image_bin = np.where(image >= threshold, 0, 1)
        image_bin = image_bin.reshape(image_bin.shape[0:2])

        features[idx, 2] = np.count_nonzero(image_bin)
        image_bin_no_holes = binary_fill_holes(image_bin)

        contours = find_contours(image_bin)
        _, exterior_contours = label(image_bin, return_num=True)

        features[idx, 3] = len(contours) - exterior_contours
        features[idx, 4] = exterior_contours

    return features


