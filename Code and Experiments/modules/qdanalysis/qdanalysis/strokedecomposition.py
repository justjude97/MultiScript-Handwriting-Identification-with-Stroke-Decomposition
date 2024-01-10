"""
strokedecomposition.py

stroke decomposition algorithms take in a preprocessed image and return a list of extracted strokes
* strokes are approximations
"""

import numpy as np
import cv2 as cv
import qdanalysis.preprocessing as prep

from skimage.morphology import skeletonize
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import find_objects


#skeleton network library
import sknw

def label_graph_edges(im_graph, im_shape):
    labels = np.zeros(shape=im_shape)

    #construct a labeled image from the graph consisting of all edges
    for label_idx, (node1, node2, idx) in enumerate(im_graph.edges):
        edge_points = im_graph[node1][node2][idx]['pts']
        labels[edge_points[:, 0], edge_points[:, 1]] = label_idx + 1 #need to account for zero indexing

    return labels

"""
takes in a image and it's resulting labels image and uses the labels to segment the original image via masking

parameters:
* image - a preprocessed iamge represented as an nd_array
* labels - an array that is the same size as image and contains integer labels representing segmented portions
    of a handwriting stroke
"""
def mask_grayscale(image, labels):
    try:
        #might consider relaxing this constraint to the row and column dimension for multi-dimensional spatial data
        if image.shape != labels.shape:
            raise ValueError("image of size " + str(image.shape) +
                              " and label array of size " + str(labels.shape) + " are not the same.")
        
    except ValueError as err:
        print(err)

"""
uses KNearestNeighbors to take a label image and a foreground image and group the non-zero pixels of that foreground image according to the closest pixel label.
* labels and foreground should be of the same size
* labels is a integer array of n different class labels
"""
def knnRegionGrowth(labels, foreground):
    #this is the "train" and "test" set for the knn classifier, what the other values are going to be matched to
    label_coords = np.transpose(labels.nonzero())
    label_vals = labels[label_coords[:, 0], label_coords[:, 1]]
    
    #knn classifier will label foreground element via closest skeleton point
    cls = KNeighborsClassifier(n_neighbors=1)
    cls.fit(label_coords, label_vals)

    #now grab the coordinates of all the foreground elements and match them to a label
    img_coords = np.transpose(foreground.nonzero())
    img_labels = cls.predict(img_coords)

    segmented_image = np.zeros_like(foreground, dtype=int)
    segmented_image[img_coords[:, 0], img_coords[:, 1]] = img_labels

    return segmented_image

"""
should probably move this to it's own file later
"""
def preprocess_skeletonize(image):
    bin_image = cv.threshold(image, 0, 1, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    return skeletonize(bin_image)


"""
baseline stroke decomposition for graph based techniques. simply skeletonizes an image and turns it into a graph to 
    extract edge segments. implicitly returns grayscale image if passed

parameters:
* image - a preprocessed image represented as an nd_array
"""
def simple_stroke_segment(image):

    #needs to be boolean for region growing, but if it's already boolean then there's no need to threhold
    image_is_bool = isinstance(image.flat[0], np.bool_)
    foreground = prep.preprocess(image) if not image_is_bool else image

    #TODO should be a part of the preprocessing
    im_skeleton = skeletonize(foreground)

    #convert to skeleton to graph representation
    #TODO: sknw doesn't build complete edges, need to fix this
    im_graph = sknw.build_sknw(im_skeleton, multi=True, full=True, ring=True)

    labels = label_graph_edges(im_graph, image.shape)
    
    segmented_image = knnRegionGrowth(labels, foreground)

    #now that we have segmented the image, we need to extract the segments as a list of individual strokes
    extracted_strokes = []

    #bounding box coords of image labels, should line up with label numbers
    stroke_coords = find_objects(segmented_image)

    for idx, bb in enumerate(stroke_coords):
        #get bounding box of segmented label and filter any other labels in that bounding box
        filter = (segmented_image[bb] == idx + 1)
        extracted_strokes.append(filter)

    return extracted_strokes



