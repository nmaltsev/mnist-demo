import numpy as np
import cPickle as pickle
import keras

from keras.callbacks import Callback

class TrainingLog(keras.callbacks.Callback):
    def __init__(self, log_path='train_batchs.data'):
        super(Callback, self).__init__()
        self.log_path = log_path
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.loss = []
        self.acc = []
        self.batch = []
        self.size = []
        self.logs = []
        
    def on_batch_end(self, batch, logs={}):
        self.logs.append(logs)
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.batch.append(logs.get('batch'))
        self.size.append(logs.get('size'))

        with open(self.log_path, 'wb') as f:
            pickle.dump({
                    'batch': self.batch,
                    'loss': self.loss,
                    'acc': self.acc,
                    'size': self.size
            }, f, pickle.HIGHEST_PROTOCOL)
