import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import keras
import cPickle as pickle

def plotChart(losses, acc, val_losses, val_acc):
    # Clear the previous plot
    # clear_output(wait=True)
    N = np.arange(0, len(losses))
    
    # You can chose the style of your preference
    # print(plt.style.available) to see the available options
    plt.style.use('seaborn')
    
    # Plot train loss, train acc, val loss and val acc against epochs passed
    plt.figure()
    plt.plot(N, losses, label = "train_loss")
    plt.plot(N, acc, label = "train_acc")
    plt.plot(N, val_losses, label = "val_loss")
    plt.plot(N, val_acc, label = "val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
    # libpng-dev must be installed
    # plt.savefig('training_plot')

class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        with open('plot.data', 'wb') as f:
            pickle.dump({
                'epoch': epoch,
                'losses': self.losses,
                'acc': self.acc,
                'val_losses': self.val_losses,
                'val_acc': self.val_acc
            }, f, pickle.HIGHEST_PROTOCOL)

    def on_train_end(self, logs={}):
        print('The training was complited')
        plotChart(self.losses, self.acc, self.val_losses, self.val_acc)
