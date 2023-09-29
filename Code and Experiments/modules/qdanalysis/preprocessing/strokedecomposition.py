"""
stroke decomposition algorithms take in a preprocessed image and return a list of extracted strokes
* strokes are approximations
"""

import numpy as np
import cv2 as cv

from skimage.morphology import skeletonize
from sklearn.neighbors import KNeighborsClassifier


#skeleton network library
import sknw

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
baseline stroke decomposition for graph based techniques. simply skeletonizes an image and turns it into a graph to 
    extract edge segments. implicitly returns grayscale image if passed

parameters:
* image - a preprocessed image represented as an nd_array
"""
def simple_stroke_segment(image):

    #needs to be boolean for region growing, but if it's already boolean then there's no need to threhold
    image_is_bool = isinstance(image.flat[0], np.bool_)
    foreground = cv.threshold(image, 0, 1, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] if image_is_bool else image

    im_skeleton = skeletonize(foreground)
    #build a graph from the skeletonized image and allow multple branches between nodes and allow loops in handwriting
    im_graph = sknw.build_sknw(im_skeleton, multi=True, full=True, ring=True)
    labels = np.zeros_like(image)

    #construct a labeled image from the graph consisting of all edges
    #TODO: sknw doesn't build complete edges, need to fix this
    for label_idx, (node1, node2, idx) in enumerate(im_graph.edges):
        edge_points = im_graph[node1][node2][idx]['pts']

        labels[edge_points[:, 0], edge_points[:, 1]] = label_idx + 1 #need to account for zero indexing
    
    #this is the "train" and "test" set for the knn classifier, what the other values are going to be matched to
    label_coords = np.array(labels.nonzero()).T
    label_vals = labels[label_coords[:, 0], label_coords[:, 1]]

    #knn classifier will label foreground element via closest skeleton point
    cls = KNeighborsClassifier(n_neighbors=1)
    cls.fit(label_coords, labels)

    #now grab the coordinates of all the foreground elements and match them to a label
    img_coords = foreground.nonzero().T
    img_labels = cls.predict(foreground.nonzero().T)

    segmented_image = np.zeros_like(image)
    segmented_image[img_coords[:, 0], img_coords[:, 1]] = img_labels

    return segmented_image



