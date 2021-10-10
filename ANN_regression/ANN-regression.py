import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


'''
for df1 dataset

RMSD-Size of the residue.
F1 - TSA  - Total surface area.
F2 - NPEA - Non polar exposed area.
F3 - Fractional area of exposed non polar residue.
F4 - Fractional area of exposed non polar part of residue.
F5 - Molecular mass weighted exposed area.
F6 - Average deviation from standard exposed area of residue.
F7 - Euclidian distance.
F8 - Secondary structure penalty.
F9 - Spacial Distribution constraints (N,K Value).
'''
print (tf.__version__)

# df = pd.read_excel("Folds5x2_pp.xlsx")
df = pd.read_csv("CASP.csv")
X = df.iloc[1:, :-1].values
y = df.iloc[1:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=8, activation= 'relu'))
ann.add(tf.keras.layers.Dense(units=8, activation= 'relu'))
ann.add(tf.keras.layers.Dense(units=1)) 
'''
if using classification with 2 outcomes like yes/no, 0/1 --> use sigmoid activation func
if using classification with >2 outcomes --> softmax
for regression we don't need activation function
'''

# compiling the ANN

ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(X_train, y_train, batch_size=32, epochs=150)

#predicting the results

y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


