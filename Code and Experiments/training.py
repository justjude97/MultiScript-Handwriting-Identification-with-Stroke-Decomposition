"""
Implement's actual model training: including checkpoints, and logging
"""
from qdanalysis.models import adapt_resnet50
import tensorflow as tf
import tensorflow.keras as keras

def train_model(root_dir: str, train_set, validation_set):

    #Callback Functions
    #Checkpoint Callback - model directory
    #Tensorboard Callback - logging directory
    
    #data loading
    label_mode = 'categorical'
    #shape image will be resized to (figure out ideal values later)
    input_shape = (128, 128, 3)
    seed = 42
    validation_split = 0.2

    #due to nature of data, the test and train splits are defined beforehand
    train_data = keras.preprocessing.image_dataset_from_directory(train_set_fp, 
                                                                label_mode=label_mode, 
                                                                image_size=input_shape[:2], 
                                                                seed=seed, 
                                                                validation_split=validation_split, 
                                                                subset='training')
    train_classes = train_data.class_names

    validation_data = keras.preprocessing.image_dataset_from_directory(train_set_fp,
                                                                   label_mode=label_mode,
                                                                   image_size=input_shape[:2],
                                                                   seed=seed,
                                                                   validation_split=validation_split,
                                                                   subset='validation')
    train_classes_val = validation_data.class_names

    #check if train and validationsets have the same writer class
    assert train_classes == train_classes_val, "training set and validation set have different labels"
    
    #Optimizer and Hyperparameters
    # optimizer and loss taken from https://stackoverflow.com/questions/71704268/using-tf-keras-utils-image-dataset-from-directory-with-label-list
    learn_rate = 1e-5
    first_momentum_decay = 0.9
    epsilon=1e-7

    optimizer = keras.optimizers.Nadam(
        learning_rate=learn_rate,
        beta_1=first_momentum_decay,
        epsilon=epsilon,
        name='Nadam'
    )

    lossfn = keras.losses.CategoricalCrossentropy()

    metrics = [
        keras.metrics.CategoricalAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall()
    ]

    #model and training
    freeze_weights = False
    model = adapt_resnet50(input_shape, len(writer_classes), False)

    