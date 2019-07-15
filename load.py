# coding: utf-8

from keras.models import model_from_json

from config import CNN_config
from model import create_MNIST_CNN
from dataset import load_dataset

# Loading the dataset
X_train, X_test, Y_train, Y_test = load_dataset()


# Download the network architecture data from the json file
json_file = open(CNN_config.model_json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create a model based on the loaded data
loaded_model = model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights(CNN_config.model_weight_path)

# Compile the model
loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# Check the model with test data
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("The accuracy on test dataset: %.2f%%" % (scores[1]*100))