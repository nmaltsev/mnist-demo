# coding: utf-8
# The objective of this script is to learn Keras early stopping feature
import sys
import numpy

sys.path.append('./../')

from config import CNN_config
from model import create_MNIST_CNN, create_MNIST_CNN1, create_MNIST_CNN2
from utils.dataset import load_dataset, getDataGen, repack_dataset
from utils.hardware import configureHardware
from utils.timer import Timer
from utils.training_plot import TrainingPlot
from utils.early_stopping_by_loss import EarlyStoppingByLoss

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################################################

# Set the seed value for repeatability
numpy.random.seed(42)
configureHardware()

########################################################################
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

model = create_MNIST_CNN()
model = create_MNIST_CNN1() # the worstest
model.compile(**COMPILE_MODES.get(2))
# print(model.summary())


########################################################################
validation_split = 0.1
# earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
# modelCheckpoint = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# Train the network
if CNN_config.use_fit_generator:
    X_train, X_test, Y_train, Y_test, validation_data = repack_dataset(validation_split, *load_dataset())
    # This will do preprocessing and realtime data augmentation:
    datagen = getDataGen()
    datagen.fit(X_train)
    plot_losses = TrainingPlot()
    timer = Timer().start()
    # earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    # modelCheckpoint = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=CNN_config.batch_size),
        len(X_train),
        CNN_config.epochs,
        validation_data=validation_data,
        callbacks=[
            plot_losses, 
            # earlyStopping, 
            # modelCheckpoint,
            EarlyStoppingByLoss(threshold=0.01)
        ]
    )
    timer.stop().note('Prediction time')
    timer.summary()
	
else:
    X_train, X_test, Y_train, Y_test = load_dataset()

    model.fit(
        X_train, 
        Y_train, 
        batch_size=CNN_config.batch_size, 
        nb_epoch=CNN_config.epochs, 
        validation_split=validation_split, 
        # verbose=2 # this will hide the progress
    )

# We evaluate the quality of network training on test data
scores = model.evaluate(X_test, Y_test, verbose=0)
print("The accuracy on test dataset: %.2f%%" % (scores[1]*100))

# Get json description of model
model_json = model.to_json()

# Save model description in file
json_file = open(CNN_config.model_json_path, 'w')
json_file.write(model_json)
json_file.close()

# Save model weights in file
model.save_weights(CNN_config.model_weight_path, overwrite=True)