#! /usr/bin/python3.6

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# data preprocessing training set

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')        # if we have >2 its categorical


# # data preprocessing test set

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# building the CNN
cnn = tf.keras.models.Sequential()

# S1: 1st convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape = [64, 64, 3]))

# S2: Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# 2nd convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) # remove the input_shape parameter
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# S3: Flattening to convert 3D tensor to 1D vector
cnn.add(tf.keras.layers.Flatten())

# S4: Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the CNN

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# training the CNN on the training set and evaluating on the test set

cnn.fit(x = training_set, validation_data= test_set, epochs=25)

# making a prediction

from tensorflow.keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size= (64, 64))
test_image = image.img_to_array(test_image)

# add extra dimension to convert to batch
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)

# encoding
training_set.class_indices

if result[0][0] == 1:   # batch and image in batch
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

