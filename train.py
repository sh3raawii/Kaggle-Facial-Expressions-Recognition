# Author: Mostafa Mahmoud Ibrahim Hassan
# Email: mostafa_mahmoud@protonmail.com

import os
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.layers.pooling import MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import rmsprop, Adam
from load_data import DataLoader


def create_model():
    """
    Creates the keras model, feel free to try alter the architecture.
    :return: Keras model
    """
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding='same', input_shape=(48, 48, 1), kernel_regularizer=l2(), activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(), activation='relu'))
    model.add(MaxPool2D(padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(), activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(), activation='relu'))
    model.add(MaxPool2D(padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(), activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(), activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=l2(), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(7, activation='softmax'))
    return model


if __name__ == '__main__':

    # Load Data, default csv path is data/fer2013.csv
    train_x, train_y, test_x, test_y = DataLoader().load_data()

    # data pre-processing
    # we can subtract the mean image
    # mean = train_x.mean(axis=0)
    # or we can subtract the mean color
    mean = np.mean(np.mean(np.mean(train_x, 2), 1), 0)
    train_x = train_x - mean
    test_x = test_x - mean

    # get user choice
    message = "Create model from scratch? [1]\nLoad a pre-trained model? [2]\n>>Please enter your choice: "
    choice = input(message)
    while not choice.isnumeric() or int(choice) < 1 or int(choice) > 2:
        choice = input(">>Please enter a valid choice: ")
    choice = int(choice)

    # switch case
    my_model = None
    if choice == 1:
        my_model = create_model()
        # set the optimizer
        # optimizer = rmsprop(0.001)
        optimizer = Adam(0.001)
        my_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    elif choice == 2:
        model_path = input("Enter model path including h5 extension [for default 'my_model.h5' press Enter]: ")
        model_path = './my_model.h5' if model_path == "" else model_path
        model_path = os.path.realpath(model_path)
        assert os.path.isfile(model_path)
        my_model = load_model(model_path)
    else:
        exit("Error: Something really weird happened\nPlease report this, invalid choice: " + str(choice))

    # Setting callbacks
    batch_size = 128
    minimum_lr = 0.00001
    lr_reduction_factor = 0.67
    lr_reduction_patience = 5
    early_stopping_patience = 50
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")
    file_path = "weights.epoch{epoch:02d}-acc{acc:.2f}-loss{loss:.2f}-val_loss{val_loss:.2f}.h5"
    saver_callback = keras.callbacks.ModelCheckpoint(os.path.join(checkpoints, file_path), monitor='val_loss',
                                                     verbose=1, save_best_only=True, mode='auto', period=1)
    monitor = keras.callbacks.TensorBoard(log_dir=os.path.join(os.path.dirname(__file__), "log"),
                                          histogram_freq=0, write_graph=False, write_images=False)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_reduction_factor,
                                                   patience=lr_reduction_patience, verbose=1, mode='auto',
                                                   epsilon=0.0001, cooldown=0, min_lr=minimum_lr)
    stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stopping_patience,
                                                      verbose=1, mode='auto')

    # Train
    train_validation_split = 0.25
    training_epochs = 150
    my_model.fit(train_x, train_y, validation_split=train_validation_split, batch_size=batch_size,
                 epochs=training_epochs, callbacks=[saver_callback, monitor, lr_reducer, stopping_callback])
    # Evaluate
    test_eval = my_model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    print("Test-Set Evaluation : ", test_eval)
    print("Finished.")
