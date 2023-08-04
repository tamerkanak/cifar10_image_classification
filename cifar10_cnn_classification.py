# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:45:29 2023

@author: tamer
"""
# Import required libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import keras
import random

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Reshape the labels to 1D arrays
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# Normalize the pixel values to be in the range [0, 1]
x_train = x_train / 255
x_test = x_test / 255

# Print the shapes of the training and testing data
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Define the class labels for CIFAR-10
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Function to display an image and its corresponding label
def displayImg(x, y, index):
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])

# Create the CNN model using Keras
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Flatten(),

    layers.Dense((64), activation="relu"),
    layers.Dense((10), activation="softmax")
])

# Compile the model with the appropriate loss function, optimizer, and metrics
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model using the training data
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model using the testing data
model.evaluate(x_test, y_test)

# Predict the labels for the testing data
y_pred = model.predict(x_test)

# Get the predicted class labels for each sample
y_preds = [np.argmax(element) for element in y_pred]

# Generate a random index to display an example from the testing data
random_predict = random.randint(0, 10000)

# Display the image and the predicted label
displayImg(x_test, y_test, random_predict)
print("The predict is: ", classes[y_preds[random_predict]])