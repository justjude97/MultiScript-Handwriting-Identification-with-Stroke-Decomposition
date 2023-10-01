"""
methods that build keras models
"""

import tensorflow as tf
import tensorflow.keras as keras

def adapt_resnet50(input_shape, freeze_weights=True):
    input = keras.layers.Input(shape=input_shape)
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights='imagenet',
                                                      input_shape=input_shape)(input)
    
    #TODO make number of layers adjustable for hyper parameters
    pooling_layer = keras.layers.GlobalAveragePooling2D()(base_model)
    hidden_1 = keras.layers.Dense(1000, activation='relu')(pooling_layer)
    output = keras.layers.Dense(10, activation='softmax')(hidden_1)

    return keras.Model(inputs=input, outputs=output)