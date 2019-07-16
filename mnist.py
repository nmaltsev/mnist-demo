# import matplotlib.pyplot as plt
import numpy as np
import keras

print('The MNIST express test for keras {}'.format(keras.__version__))

def parse_version(version_s):
    return [int(n) for n in version_s.split('.')]

kver_n = parse_version(keras.__version__)[0]

def get_dataset(kver_n):
    from keras.datasets import mnist
    
    if kver_n == 2:
        def to_categorical(array, nb_classes):
            return keras.utils.to_categorical(array, num_classes=nb_classes)
    elif kver_n == 1:
        from keras.utils.np_utils import to_categorical
    else:
        print('Unsupported keras version')
        exit(0)
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Data normalization
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test        

def get_model():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    
    """
    Here, the neural network has been defined as a sequence of two layers that are densely connected (or fully connected), meaning that all the neurons in each layer are connected to all the neurons in the next layer.
     the loss function that we will use to evaluate the degree of error between calculated outputs and the desired outputs of the training data
    """
    
    model = Sequential()
    model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics = ['accuracy']
    )
    return model
    
x_train, y_train, x_test, y_test = get_dataset(kver_n)
# show image from dataset
# plt.imshow(x_train[8], cmap=plt.cm.binary)

kw_fit_args = {'batch_size': 128}

if kver_n == 2:
    kw_fit_args['epochs'] = 5
elif kver_n == 1:
    kw_fit_args['nb_epoch'] = 5
else:
    print('Unsupported version of keras')
    exit(0)
    
model = get_model()
model.fit(x_train, y_train, **kw_fit_args)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)
predictions = model.predict(x_test)
prediction = np.argmax(predictions[11])
print(predictions[11])
print('Prediction: ', prediction)

if test_acc > 0.5:
    print('[The MNIST test passed]')
