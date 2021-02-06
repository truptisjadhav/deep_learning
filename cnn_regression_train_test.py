from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
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

x=df1.loc[0:,col].values
y=df1.loc[0:,["y"]].values

x=x.reshape(-1,x.shape[1],1)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3,random_state=42)

model = Sequential()
model.add(Convolution1D(40, 3, input_shape=(x.shape[1], 1), activation='relu',strides=1))
#model.add(Convolution1D(20, 5, input_shape=(x.shape[1], 1), activation='relu'))
model.add(MaxPooling1D(pool_size=2,strides=1))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss="mse",optimizer="adam")
model.fit(train_x,train_y,validation_data=(test_x,test_y),epochs=10,batch_size=200)

testPredict = model.predict(test_x)
testPredict = testPredict.reshape(-1,1)

from sklearn.metrics import mean_squared_error,mean_absolute_error 
import math
print(math.sqrt(mean_squared_error(test_y,testPredict)))
print(mean_absolute_error(test_y,testPredict))