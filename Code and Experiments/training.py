"""
Implement's actual model training: including checkpoints, and logging
"""
from qdanalysis.models import adapt_resnet50
import tensorflow as tf
import tensorflow.keras as keras

def train_model(root_dir: str, writer_classes, train_set, validation_set):

    #Callback Functions
    #Checkpoint Callback - model directory
    #Tensorboard Callback - logging directory

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
    model.compile(optimizer=optimizer, loss=lossfn, metrics=metrics)

    epochs = 1
    history = model.fit(train_data, epochs=epochs, validation_data=validation_data)

if __name__ == '__main__':
    import sys
    import os

    print("starting experiment")

    #seed for validation splits
    seed = int.from_bytes(os.urandom(4), 'big')
    print("experiment seed: ", seed)

    args = sys.argv
    
    #output of experiment
    experiment_dir = args[1]
    print("model weights, logs, and tensorboard files will be stored in ", experiment_dir)
    #dataset path should be second parameter
    dataset_fp = args[2]

    if len(args) == 4:
        print("training with no merging")

        #specified script should be 2 (and above for the merge case)
        script = args[3]

        train_path = os.path.join(dataset_fp, script)

        print("attempting to load ", train_path)
        #data loading
        label_mode = 'categorical'
        #shape image will be resized to (figure out ideal values later)
        input_shape = (128, 128, 3)
        validation_split = 0.2

        #data loading - standard validation split
        #due to nature of data, the test and train splits are defined beforehand
        train_data = keras.preprocessing.image_dataset_from_directory(train_path, 
                                                                    label_mode=label_mode, 
                                                                    image_size=input_shape[:2], 
                                                                    seed=seed, 
                                                                    validation_split=validation_split, 
                                                                    subset='training')
        train_classes = train_data.class_names

        validation_data = keras.preprocessing.image_dataset_from_directory(train_path,
                                                                    label_mode=label_mode,
                                                                    image_size=input_shape[:2],
                                                                    seed=seed,
                                                                    validation_split=validation_split,
                                                                    subset='validation')
        train_classes_val = validation_data.class_names

        #check if train and validationsets have the same writer class
        assert train_classes == train_classes_val, "training set and validation set have different labels"

        print("classes: ", train_classes)

        print("\n") # for clarity in log file
        
        train_model(experiment_dir, train_classes, train_data, validation_data)

    elif len(args) > 4:
        #TODO
        print("training with merging")
        print(args)
    else:
        print('no dataset file arguments or script descriptors', file=sys.stderr)