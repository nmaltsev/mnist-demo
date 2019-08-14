from keras.callbacks import Callback

# https://towardsdatascience.com/resuming-a-training-process-with-keras-3e93152ee11a

class EarlyStoppingByLoss(Callback):
    # val_loss
    """
    def __init__(self, monitor='loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
    """
    def __init__(self, threshold=0.03):
        super(Callback, self).__init__()
        self.threshold = threshold

    def on_batch_end(self, batch, logs={}):
        if logs.get('loss') <= self.threshold:
                self.model.stop_training = True
