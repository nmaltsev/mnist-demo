import os, sys, time
import numpy as np

import keras
from keras.utils import np_utils
from keras.datasets import mnist
# ~ from keras.utils

def read_file_chunk(fname, chunksize, nb_classes):
    lines = []
    while True:
        with open(fname, 'r') as myfile:
            for i, line in enumerate(myfile):
                new_line = [float(val) for val in line.split(',')]
                
                lines.append(new_line)
                if i > 0 and (i+1) % chunksize == 0:
                    lines = np.array(lines)
                    X = lines[:,:-1]
                    X = X.reshape(X.shape[0], 1,28,28)
                    print('X')
                    print(X)
                    # ~ exit(0)
                    
                    y = lines[:,-1]
                    # ~ y = np_utils.to_categorical(y, nb_classes)
                    
                    
                    yield (X, y)
                    lines = [] # resets the lines list

def exist_file(f):
    if not os.path.isfile(f):
        print('file: \"{}\\" does not exist'.format(f))
        return False
    return True

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
