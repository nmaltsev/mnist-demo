import sys
import numpy as np

from config import config
from utils.hardware import configureHardware
from utils.timer import Timer
from utils.early_stopping_by_loss import EarlyStoppingByLoss
from prepare_dataset2 import restore_dataset

from models import create_MNIST_CNN, COMPILE_MODES
from keras.preprocessing.image import ImageDataGenerator

# Set the seed value for repeatability
np.random.seed(42)
configureHardware(num_GPU=1)

model = create_MNIST_CNN()
model.compile(**COMPILE_MODES.get(0))

def splitOnParts(x_train, y_train, numberOfChunks):
    length = len(x_train)
    chunkLen = length // numberOfChunks if length % numberOfChunks == 0 else (length // numberOfChunks + 1)
    
    while 1:
        pos = 0
        n = 0
        while pos < length:
            # yield (n, pos, pos + chunkLen if pos + chunkLen < length else length)
            end = pos + chunkLen if pos + chunkLen < length else length
            yield (x_train[pos:end,], y_train[pos:end,])
            pos += chunkLen
            n += 1
        

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

if __name__ == '__main__':
    dataset = restore_dataset()
    print('Start training')

    # for chunk in splitOnParts(dataset['X_train'], dataset['Y_train'], 600):
    #     print('chunk')
    #     print(np.array(chunk[0]).shape, np.array(chunk[1]).shape)
    #     print(chunk[0])

    # exit(0)
    if config.use_fit_generator:
        if False:
            datagen = getDataGen()
            datagen.fit(dataset['X_train'])
            model.fit_generator(
                datagen.flow(dataset['X_train'], dataset['Y_train'], batch_size=config.batch_size),
                len(dataset['X_train']),
                config.epochs,
                validation_data=(dataset['X_valid'], dataset['Y_valid']),
                callbacks=[]
            )
            
        else:
            model.fit_generator(
                splitOnParts(dataset['X_train'], dataset['Y_train'], 600),
                len(dataset['X_train']),
                config.epochs,
                validation_data=(dataset['X_valid'], dataset['Y_valid']),
                callbacks=[
                    EarlyStoppingByLoss(0.4)
                ]
            )
            score = model.evaluate_generator(splitOnParts(dataset['X_test'], dataset['Y_test'], 600), len(dataset['X_test'])//600)
            print('Evaluation')
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    else :
        validation_split = 0.1
        model.fit(
            dataset['X_train'], 
            dataset['Y_train'],
            batch_size=config.batch_size, 
            nb_epoch=config.epochs, 
            validation_split=validation_split, 
            # verbose=2 # this will hide the progress
        )
