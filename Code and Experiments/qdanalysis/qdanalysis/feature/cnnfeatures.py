import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from qdanalysis.feature.featuretypes import featureTypes

#architecture for one half of the CNN model
#may not be needed. kept for rebuilding purposes
def halfCNNArch(inp_img):
    model = inp_img

    model = layers.Conv2D(32, kernel_size=(3, 3), activation = 'relu', padding = 'valid')(model)
    model = layers.MaxPooling2D((2, 2), padding='valid')(model)
    model = layers.Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'valid')(model)
    model = layers.MaxPool2D((2, 2), padding = 'valid')(model)
    model = layers.Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'valid')(model)
    model = layers.MaxPool2D((2, 2), padding = 'valid')(model)
    model = layers.Conv2D(256, kernel_size=(1, 1), activation = 'relu', padding = 'valid')(model)
    model = layers.MaxPool2D((2, 2), padding = 'valid')(model)
    model = layers.Conv2D(64, kernel_size=(1, 1), activation = 'relu', padding = 'valid')(model)
    model = layers.Flatten()(model)
    
    return model


def fullCNNArch(imShape):
    """
        architecture for the full CNN model without SIFT descriptors
    """
    inpImg = layers.Input(shape = imShape, name = 'ImageInput')

    model = halfCNNArch(inpImg)

    #feat is the final model, labeled, with the imputs and outputs specified
    feat = models.Model(inputs = [inpImg], outputs = [model], name = 'Feat_Model')

    #define the entry point for the input images
    left_img = layers.Input(shape = imShape, name = 'left_img')
    right_img = layers.Input(shape = imShape, name = 'right_img')

    #the weights are reused in both instances since feat is a model
    left_feats = feat(left_img)
    right_feats = feat(right_img)

    merged_feats = layers.concatenate([left_feats, right_feats], name = 'subtracted_feats')

    merged_feats = layers.Dense(1024, activation = 'linear')(merged_feats)
    #https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
    #https://keras.io/api/layers/normalization_layers/batch_normalization/
    #this layer normalizes its input; It appliesa  trainformation that maintain the mean output close to 0,
    #    and the output stddev close to 1
    merged_feats = layers.BatchNormalization()(merged_feats)
    merged_feats = layers.Activation('relu')(merged_feats)
    merged_feats = layers.Dense(4, activation = 'linear')(merged_feats)
    merged_feats = layers.BatchNormalization()(merged_feats)
    merged_feats = layers.Activation('relu')(merged_feats)
    merged_feats = layers.Dense(1, activation = 'sigmoid')(merged_feats)

    #now combine the feature models with the merged_feats model
    similarity_model = models.Model(inputs = [left_img, right_img], outputs = [merged_feats], name = 'Similarity_Model')

    return similarity_model

#TODO rename
def CNNSiftConfigOne(imShape, num_descriptors):
    """
        CNN architecture using the absolute difference of SIFT descriptors

        parameters:
        * imshape - the shape of the input images
        * num_descriptors - the number of descriptors to use in the model
    """
    inpImg = layers.Input(shape = imShape, name='ImageInput')

    halfCNN = halfCNNArch(inpImg)

    feat = models.Model(inputs=[inpImg], outputs=[halfCNN], name='cnn_model')
    left_img = layers.Input(shape = imShape, name = 'left_img')
    right_img = layers.Input(shape = imShape, name = 'right_img')
    left_feats = feat(left_img)
    right_feats = feat(right_img)
    descriptor_size = 128
    descriptors = layers.Input(shape=(descriptor_size*num_descriptors), name="image_descriptors")

    merged_feats = layers.concatenate([left_feats, right_feats, descriptors])

    #    and the output stddev close to 1
    merged_feats = layers.BatchNormalization()(merged_feats)
    merged_feats = layers.Activation('relu')(merged_feats)
    merged_feats = layers.Dense(1024, activation = 'relu')(merged_feats)
    merged_feats = layers.Dense(512, activation = 'relu')(merged_feats)
    merged_feats = layers.Dense(256, activation = 'relu')(merged_feats)
    merged_feats = layers.Dense(4, activation = 'relu')(merged_feats)
    merged_feats = layers.Dense(1, activation = 'sigmoid')(merged_feats)

    #now combine the feature models with the merged_feats model
    similarity_model = models.Model(inputs = [left_img, right_img, descriptors], outputs = [merged_feats], name = 'CNN_SIFT_model_config_1')

    return similarity_model

def CNNSiftConfigTwo(im_shape, num_descriptors):
    """
        CNN architecture using the difference of l1 norm of SIFT descriptors

        parameters:
        * imshape - the shape of the input images
        * num_descriptors - the number of descriptors to use in the model
    """
    inpImg = layers.Input(shape = im_shape, name='ImageInput')

    halfCNN = halfCNNArch(inpImg)

    feat = models.Model(inputs=[inpImg], outputs=[halfCNN], name='cnn_model')
    left_img = layers.Input(shape = im_shape, name = 'left_img')
    right_img = layers.Input(shape = im_shape, name = 'right_img')
    left_feats = feat(left_img)
    right_feats = feat(right_img)
    descriptors = layers.Input(shape=(num_descriptors), name="image_descriptors")

    cnn_diff = layers.subtract(inputs=[left_feats, right_feats], name = "difference_of_CNN_features")
    cnn_norm = tf.norm(cnn_diff, axis=1, keepdims=True)

    #return models.Model(inputs=[left_img, right_img], outputs=[cnn_norm], name="debugging")
    
    merged_feats = layers.concatenate([cnn_norm, descriptors])

    #    and the output stddev close to 1
    merged_feats = layers.BatchNormalization()(merged_feats)
    merged_feats = layers.Activation('relu')(merged_feats)
    merged_feats = layers.Dense(4, activation = 'linear')(merged_feats)
    merged_feats = layers.BatchNormalization()(merged_feats)
    merged_feats = layers.Activation('relu')(merged_feats)
    merged_feats = layers.Dense(1, activation = 'sigmoid')(merged_feats)
    #now combine the feature models with the merged_feats model
    similarity_model = models.Model(inputs = [left_img, right_img, descriptors], outputs = [merged_feats], name = 'CNN_SIFT_model_config_1')

    return similarity_model





#pretrained model for pairs of images of size 64x64
class cnnModel:

    def __init__(self, modelDir = "qdanalysis/feature/models/siamese_cnn"):
        try:
            self.model = models.load_model(modelDir)
        except BaseException as err:
            self.model = None
            print(f"Unexpected {err=}, {type(err)=}")
            return
        
        self.type = featureTypes.CNN

    #TODO: wont be able to work with the feature extraction pipeline as normal. try and resolve.
    def predict(self, imageCmp):
        predictions = self.model.predict(imageCmp)
        return (self.type, predictions)

#pretrained model for pairs of images of size 64x64
class halfCnnModel:

    def __init__(self, modelDir = "qdanalysis/feature/models/half_siamese_cnn"):
        try:
            self.model = models.load_model(modelDir)
        except BaseException as err:
            self.model = None
            print(f"Unexpected {err=}, {type(err)=}")
            return

        self.type = featureTypes.CNNHALF

    def extract(self, images: np.ndarray):
        predictions = self.model.predict(images)
        return (self.type, predictions)

    
