"""
stroke decomposition algorithms take in a preprocessed image and return a list of extracted strokes
* strokes are approximations
"""

import cv2 as cv
from skimage.morphology import skeletonize
import numpy as np

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
    extract edge segments

parameters:
* image - a preprocessed image represented as an nd_array
"""
def simple_stroke_segment(image, grayscale=True):
    #TODO add checking for binary image and convert if not

    im_skeleton = skeletonize(image)
    #build a graph from the skeletonized image and allow multple branches between nodes and allow loops in handwriting
    im_graph = sknw.build_sknw(image, multi=True, full=True, ring=True)
    labels = np.zeros_like(image)

    #construct a labeled image from the graph consisting of all edges
    #TODO: sknw doesn't build complete edges, need to fix this
    for label_idx, (node1, node2, idx) in enumerate(graph.edges):
        edge_points = im_graph[node1][node2][idx]['pts']

        labels[edge_points[:, 0], edge_points[:, 1]] = label_idx + 1 #need to account for zero indexing
    
    if grayscale:
        return mask_grayscale(image, labels)



