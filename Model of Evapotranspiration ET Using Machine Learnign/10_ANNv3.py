# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 21:10:09 2018
@author: Jose
A Neural Network, used Keras (TensorFlow backend) to 
Classify the ET data with Tmedia, Urmedia, Rad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Importing the dataset
dataset = pd.read_csv("ETo.csv")
X = dataset[['Tmedia', 'Urmedia', 'Rad']].copy()
#X = dataset.filter(['Tmedia', 'Urmedia', 'Rad'],axis=1)

y_ = dataset.iloc[:,9].values
y_ = y_.astype(np.int32)
y_ = y_.reshape(-1, 1) # Convert data to a single column

# One Hot encode the class labels
x = X

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)
#print(y)

# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

# Build the model

model = Sequential()

model.add(Dense(10, input_shape=(3,), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(4, activation='softmax', name='output'))

# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

# Train the model
model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=300)

# Test on unseen data

results = model.evaluate(test_x, test_y)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))