from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential

nb_filters = 32                         # number of convolutional filters to use
nb_pool = 2                             # size of pooling area for max pooling
nb_conv = 3   #3                        # convolution kernel size

class ModelProperties_MNIST:
    nb_classes = 10
    img_rows, img_cols = 28, 28 # input image dimensions
    img_channel = 1

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
        input_shape=(ModelProperties_MNIST.img_channel, ModelProperties_MNIST.img_rows, ModelProperties_MNIST.img_cols)
        # input_shape= (ModelProperties_MNIST.img_channel, ModelProperties_MNIST.img_rows * ModelProperties_MNIST.img_cols)
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