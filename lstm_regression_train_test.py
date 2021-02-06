from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, LSTM,TimeDistributed
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D

import numpy as np
import pandas as pd

df=pd.read_csv("/data/regression.csv")
df1=df[df.y < 47]

col=df.columns.tolist()
col.remove("id")
col.remove("y")
col = col[180:195]
x=df1.loc[0:,col].values
y=df1.loc[0:,["y"]].values

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler
scaler_x    = MinMaxScaler().fit(train_x)
train_x_sc  = scaler_x.transform(train_x)
test_x_sc   = scaler_x.transform(test_x)

scaler_y    = MaxAbsScaler().fit(train_y)
train_y_sc  = scaler_y.transform(train_y)
test_y_sc   = scaler_y.transform(test_y)

train_x_sc = train_x_sc.reshape(train_x_sc.shape[0],train_x_sc.shape[1],1)
test_x_sc  = test_x_sc.reshape(test_x_sc.shape[0],test_x_sc.shape[1],1)


model = Sequential()
model.add(LSTM(20, input_shape=(train_x.shape[1], 1)))
model.add(Dense(1))
model.compile(loss="mse",optimizer="adam")
model.fit(train_x_sc,train_y_sc,validation_data=(test_x_sc,test_y_sc),epochs=10,batch_size=1)

import math
from sklearn.metrics import mean_squared_error
testPredict = model.predict(test_x_sc)
testY1 = scaler_y.inverse_transform(testPredict)

testScore = math.sqrt(mean_squared_error(test_y[:,0], testY1[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

##########
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(train_x.shape[1], 1)))
model.add(TimeDistributed(Dense(1)))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss="mse",optimizer="adam")
model.fit(train_x_sc,train_y_sc,validation_data=(test_x_sc,test_y_sc),epochs=10,batch_size=100)

testPredict = model.predict(test_x_sc)
testY1 = scaler_y.inverse_transform(testPredict)

testScore = math.sqrt(mean_squared_error(test_y[:,0], testY1[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

