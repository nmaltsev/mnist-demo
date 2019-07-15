import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils

from config import CNN_config
from model import ModelProperties_MNIST

from keras.preprocessing.image import ImageDataGenerator


def load_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], ModelProperties_MNIST.img_channel, ModelProperties_MNIST.img_rows, ModelProperties_MNIST.img_cols)
    X_test = X_test.reshape(X_test.shape[0], ModelProperties_MNIST.img_channel, ModelProperties_MNIST.img_rows, ModelProperties_MNIST.img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, ModelProperties_MNIST.nb_classes)
    Y_test = np_utils.to_categorical(y_test, ModelProperties_MNIST.nb_classes)
    return X_train, X_test, Y_train, Y_test

def getDataGen():
    return ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
    )

def repack_dataset(validation_split_n, x_train, x_test, y_train, y_test):
    validation_size = int(len(x_train) * validation_split_n)
    index = np.random.permutation(len(x_train))
    train_index = index[:-validation_size]
    valid_index = index[-validation_size:]

    return x_train[train_index], x_test, y_train[train_index], y_test, (x_train[valid_index], y_train[valid_index])


