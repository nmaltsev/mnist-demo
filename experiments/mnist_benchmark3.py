import sys
import numpy as np
import keras

sys.path.append('./../')
from utils.early_stopping_by_loss import EarlyStoppingByLoss
from utils.training_log import TrainingLog
from utils.timer import Timer
from utils.hardware import configureHardware

# Set the seed value for repeatability
np.random.seed(42)
configureHardware()

def parse_version(version_s):
    return [int(n) for n in version_s.split('.')]

kver_n = parse_version(keras.__version__)[0]
EPOCH_N = 20
print('The MNIST benchmark for Keras {}'.format(keras.__version__))


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
    
x_train, y_train, x_test, y_test = get_dataset(kver_n)
# show image from dataset
# plt.imshow(x_train[8], cmap=plt.cm.binary)

kw_fit_args = {
    'batch_size': 128,
    'callbacks': [
        EarlyStoppingByLoss(threshold=0.012),
        TrainingLog(log_path='mnist_fc.log')
    ]
}

if kver_n == 2:
    kw_fit_args['epochs'] = EPOCH_N
elif kver_n == 1:
    kw_fit_args['nb_epoch'] = EPOCH_N
else:
    print('Unsupported version of keras')
    exit(0)

model = get_model()
model.compile(**COMPILE_MODES.get(2))
# model.summary()
timer = Timer().start()    
model.fit(x_train, y_train, **kw_fit_args)
timer.stop().note('Prediction time')
timer.summary()
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy: {}, loss: {}'.format(test_acc, test_loss))
predictions = model.predict(x_test)
prediction = np.argmax(predictions[11])
print(predictions[11])
print('Prediction: ', prediction)

if test_acc > 0.5:
    print('[The MNIST test passed]')
