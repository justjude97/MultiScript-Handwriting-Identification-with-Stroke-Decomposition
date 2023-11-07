from qdanalysis.feature.featuretypes import featureTypes
import cv2 as cv
import numpy as np

class siftExtractor:
    
    def __init__(self):
        self.sift = cv.SIFT_create()

        self.type = featureTypes.SIFT

    #images should be a numpy array of size(-1, imgsz, imgsz, 1)
    # different images along first axis 
    def extract(self, image: np.ndarray):

        #assume that the images are a normalized float and convert to 
        if(image.dtype != np.uint8):
            image = np.multiply(image, 255).astype(np.uint8)

        descriptors = []
        for img in image:
            _, des = self.sift.detectAndcompute(image[img], None)
            descriptors.append(des)

        return (self.type, descriptors)
