#imports
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense
from sklearn.metrics import classification_report
import numpy as np

train_data = pd.read_csv("air_quality_train.csv")
control_data = pd.read_csv("air_quality_test.csv")

#printing columns and their types
print(train_data.info())
#printing class distribution
print(Counter(train_data["Air_Quality"]))
#extracting features and column labels
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
y_train = train_data["Air_Quality"]
x_control = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
y_control = test_data["Air_Quality"]

#encoding the labels -> int
le = LabelEncoder()
#encoded int labels -> binary vectors
y_control=le.transform(y_test.astype(str))
y_train = tensorflow.keras.utils.to_categorical(y_train, dtype = 'int64')
y_control = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')


model = Sequential()
#input layer
model.add(InputLayer(input_shape=(x_train.shape[1],)))
#hidden layer
model.add(Dense(10, activation='relu'))
#output layer
model.add(Dense(6, activation='softmax'))

#compilation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#training and evaluation 
model.fit(x_train, y_train, epochs = 30, batch_size = 16, verbose = 0)

#stats
y_estimate = model.predict(x_control)
y_estimate = np.argmax(y_estimate, axis = 1)
y_true = np.argmax(y_control, axis = 1)
print(classification_report(y_true, y_estimate))

