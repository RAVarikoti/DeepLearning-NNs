#! /usr/bin/python3.6

import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import metrics
import tensorflow as tf

#print(tf.__version__)
#print(np.__version__)

df = pd.read_csv("Churn_Modelling.csv")
x = df.iloc[:, 3:-1].values   # removing the columns that are not important (row #, customer ID and surname)
y = df.iloc[:, -1].values

#print(x)
#print(y)

# encoding categorical data --- changing the words/labels to numerical

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2]) # encoding gender column -- male = 1 and female =0 since we have only two variables

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=0)


# feature scaling is important for the ANN, we will apply it for all variables

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

###################################################### ANN #########################################################
# initializing ANN

ann = tf.keras.models.Sequential()

# adding an input layer 1st hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # first layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # second layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # output layer --- since its binary units = 1 if you have >2 use OneHotEncoder and activation = softmax

# compiling the ANN

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics= ['accuracy']) 
# optimizer updates the weights through Stochastic gradi optimizer, loss --> categorical
# binary_crossentropy: Computes the cross-entropy loss between true labels and predicted labels.


# training the ANN on training set

ann.fit(x_train, y_train, batch_size = 32, epochs = 50)

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5) #[[]] --> i/p of predict method should be a 2d array

#predicting test results

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test, y_pred)
print(ac)

