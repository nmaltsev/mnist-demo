import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential

nb_filters = 32                         # number of convolutional filters to use
nb_pool = 2                             # size of pooling area for max pooling
nb_conv = 3                             # convolution kernel size

class ModelProperties_MNIST:
    nb_classes = 10
    input_shape = (1, 28, 28) # chanel, n_rows, n_cols


def create_MNIST_CNN():
    # Create sequential model
    model = Sequential()

    # Add network layers
    model.add(Convolution2D(
        nb_filters, 
        nb_conv, 
        nb_conv,
        init='he_normal', 
        border_mode='valid',
        input_shape=ModelProperties_MNIST.input_shape
    ))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(ModelProperties_MNIST.nb_classes))
    model.add(Activation('softmax'))    
    return model

COMPILE_MODES = {
    0: {
        'optimizer': 'SGD',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy']
    },
    1: {
        'optimizer': keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.002, nesterov=True),
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy']
    },
    2: {
        'optimizer': keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        'loss': 'mse',
        'metrics': ['accuracy']
    }
}
