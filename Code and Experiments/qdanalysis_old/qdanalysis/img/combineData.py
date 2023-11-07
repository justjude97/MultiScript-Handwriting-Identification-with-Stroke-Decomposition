import pandas as pd
import numpy as np
import cv2 as cv

def pairImages(annotations, images, mappings):
    """
        pairs a set of images given by annotations

        output is three lists:
        1. left_image
        2. right_images
        3. label

        NOTE: returns a list because the images may not be of the same size
    """
    left_images = []
    right_images = []
    labels = annotations["label"].tolist()

    #number of image pairs in the anotations file
    numEntries = len(annotations)
    for idx in range(len(annotations)):
        entry = annotations.iloc[idx]

        left_idx = mappings[entry['leftIm']]
        left_img = images[left_idx]
        right_idx = mappings[entry['rightIm']]
        right_img = images[right_idx]

        left_images.append(left_img)
        right_images.append(right_img)

    return left_images, right_images, labels

def combineDescriptors(annotations, descriptors, mappings, num_descriptors, matchType = "absdiff"):
    """
        gets the absolute difference of the n nearest descriptors

        parameters:
        * annotations - dataframe containing
        * descriptors - array of descriptors, pre computed
        * mappings - 

        returns:
        * descriptorList - an ndarray of shape
    """

    descriptorComb = None

    if(matchType == "absdiff"):
        descriptorComb = np.empty(shape=(len(annotations), num_descriptors * 128))
    elif(matchType == "diff_norm"):
        descriptorComb = np.empty(shape=(len(annotations), num_descriptors))
    else:
        print("error: matchType not known")
        return

    descriptorComb[:] = np.NaN

    #TODO definetely a better way to do this
    bf = cv.BFMatcher()
    for idx in range(len(annotations)):
        entry = annotations.iloc[idx]

        left_idx = mappings[entry['leftIm']]
        left_des = descriptors[left_idx]
        right_idx = mappings[entry['rightIm']]
        right_des = descriptors[right_idx]

        #there must be at least one descriptor for knn match to work
        #NOTE currently knn match is guaranteed to match n types together. the same descriptor could be matched multiple
        #   times. this could be a problem
        if left_des.shape[0] != 0 and right_des.shape[0] != 0:
            matches = bf.knnMatch(left_des, right_des, k=2)
            bestMatches = [m[0] for m in matches]
            bestMatches.sort(key=lambda x: x.distance)

            bestMatches = bestMatches[:num_descriptors]

            for desIdx, match in enumerate(bestMatches):
                des1 = left_des[match.queryIdx]
                des2 = right_des[match.trainIdx]
                
                if matchType == 'absdiff':
                    desDiff = cv.absdiff(des1, des2)
                    start = desIdx*128
                    end = desIdx*128 + 128
                    descriptorComb[idx, start:end] = desDiff.reshape(128,)
                elif matchType == 'diff_norm':
                    desNorm = np.linalg.norm(des1 - des2)
                    descriptorComb[idx, desIdx] = desNorm


    return descriptorComb






    descriptorList = np.ndarray
def descriptorDiff(leftDes, rightDes, numDescriptors: int):
    """
        takes the absolute difference of sift descriptors, individually

        parameters:
        * leftDes - descriptors of the first image in the comparision
        * rightDes - descriptors of the second image in the comparision
        * numDescriptors - the number of descriptor matches that we want

        returns:
        * descriptors - an ndarray of shape (n, 128)

        note - if either array is smaller than the number of requested descriptors then the "missing" descriptors
            will have the value of np.nan
    """

    #create an array of empty values. values left NaN may be filled later
    descriptors = np.empty((numDescriptors, 128))
    descriptors[:] = np.nan

    #TODO definetely a better way to do this
    bf = cv.BFMatcher()
    matches = bf.knnMatch(leftDes, rightDes, k=2)
    bestMatches = [m[0] for m in matches]
    bestMatches.sort(key=lambda x: x.distance)

    bestMatches = bestMatches[:numDescriptors]

    for desIdx, match in enumerate(bestMatches):
        des1 = leftDes[match.queryIdx]
        des2 = rightDes[match.trainIdx]

        descriptors[desIdx] = cv.absdiff(des1, des2).flatten()

def descriptorNorm(leftDes, rightDes, numDescriptors: int, ord = 1):
    """
        takes the norm of the sift descritpors

        parameters:
        * leftDes - descriptors of the first image in the comparision
        * rightDes - descriptors of the second image in the comparision
        * numDescriptors - the number of descriptor matches that we want
        * ord - the order of the norm (L1 norm default)

        returns:
        * descriptors - an ndarray of shape 

        note - if either array is smaller than the number of requested descriptors then the "missing" descriptors
            will have the value of np.nan
    """

     #create an array of empty values. values left NaN may be filled later
    descriptors = np.empty((numDescriptors,))
    descriptors[:] = np.nan

    #TODO definetely a better way to do this
    bf = cv.BFMatcher()
    matches = bf.knnMatch(leftDes, rightDes, k=2)
    bestMatches = [m[0] for m in matches]
    bestMatches.sort(key=lambda x: x.distance)

    bestMatches = bestMatches[:numDescriptors]

    for desIdx, match in enumerate(bestMatches):
        des1 = leftDes[match.queryIdx]
        des2 = rightDes[match.trainIdx]

        descriptors[desIdx] = np.linalg.norm(des1 - des2, ord = ord)

