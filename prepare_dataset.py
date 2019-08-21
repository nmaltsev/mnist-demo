import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils

from config import config
from models import ModelProperties_MNIST

def load_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], *ModelProperties_MNIST.input_shape)
    X_test = X_test.reshape(X_test.shape[0], *ModelProperties_MNIST.input_shape)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, ModelProperties_MNIST.nb_classes)
    Y_test = np_utils.to_categorical(y_test, ModelProperties_MNIST.nb_classes)
    return X_train, X_test, Y_train, Y_test
    
def repack_dataset(validation_split_n, x_train, x_test, y_train, y_test):
    validation_size = int(len(x_train) * validation_split_n)
    index = np.random.permutation(len(x_train))
    train_index = index[:-validation_size]
    valid_index = index[-validation_size:]

    return x_train[train_index], x_test, y_train[train_index], y_test, x_train[valid_index], y_train[valid_index]

def backup_dataset():
    """
    save dataset into memmap
    """
    validation_split = 0.1
    X_train, X_test, Y_train, Y_test, X_valid, Y_valid = repack_dataset(validation_split, *load_dataset())
    
    print([(np.array(array)).shape for array in [X_train, X_test, Y_train, Y_test, X_valid, Y_valid]])
    
    queue = [
        ('artefacts/x_train.m', X_train),
        ('artefacts/x_test.m', X_test),
        ('artefacts/y_train.m', Y_train),
        ('artefacts/y_test.m', Y_test),
        ('artefacts/x_valid.m', X_valid),
        ('artefacts/y_valid.m', Y_valid),
    ]
    for d in queue:
        buf = np.array(d[1])
        np.memmap(d[0], dtype='float32', mode='w+', shape=buf.shape)[:]=buf[:]
    
    
def restore_dataset():
    """
    restore dataset from memmap
    """
    dataset = {}
    queue = [
        ('artefacts/x_train.m', (54000, 1, 28, 28), 'X_train'),
        ('artefacts/x_test.m', (10000, 1, 28, 28), 'X_test'),
        ('artefacts/y_train.m', (54000, 10), 'Y_train'),
        ('artefacts/y_test.m', (10000, 10), 'Y_test'),
        ('artefacts/x_valid.m', (6000, 1, 28, 28), 'X_valid'),
        ('artefacts/y_valid.m', (6000, 10), 'Y_valid'),
    ]
    for d in queue:
        dataset[d[2]] = np.memmap(d[0], dtype='float32', mode='c', shape=d[1])
    
    return dataset

if __name__ == '__main__':
    backup_dataset()
    print('Start')
