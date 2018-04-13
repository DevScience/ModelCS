# -*- coding: utf-8 -*-
"""
Created on Fri March  30 12:05:33 2018
LinearRegression
@author: Jose
"""
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the ET
et_data = pd.read_csv("ETo.csv")

X = et_data.iloc[:,1:8].values
y = et_data.iloc[:,8].values    

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
#diference = y_test - y_pred

# Compute confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
y_test = y_test.astype(np.int32)
y_pred = y_pred.astype(np.int32)

cm = confusion_matrix(y_test, y_pred)
#plt.imshow(cm, cmap='binary', interpolation='None')

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#Accuracy score
from sklearn.metrics import accuracy_score
print('Accuracy Score: ', accuracy_score(y_test, y_pred))
