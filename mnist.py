# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:40:52 2020

@author: 91989
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Importing the dataset
data_train = pd.read_csv('fashion-mnist_train.csv')
data_test = pd.read_csv('fashion-mnist_test.csv')

trainX = data_train.iloc[:, 1:785].values
trainY = data_train.iloc[:, 0].values

testX =data_test.iloc[:, 1:785].values
testY =data_test.iloc[:, 0].values

trainX= trainX.reshape((trainX.shape[0],28,28,1))
testX= testX.reshape((testX.shape[0],28,28,1))

testX= testX.astype('float32')
trainX= trainX.astype('float32')

trainX = trainX / 255.0
testX = testX / 255.0

testY = tf.keras.utils.to_categorical(testY)
trainY = tf.keras.utils.to_categorical(trainY)


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(output_dim = 100, activation = 'relu'))
model.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(trainX, trainY,epochs = 10,batch_size=32,validation_data = (testX, testY), verbose=0)
model.summary()
_, acc = model.evaluate(testX, testY, verbose=0)
acc*100