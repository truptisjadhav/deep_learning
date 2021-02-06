import numpy as np
import pandas as pd

df=pd.read_csv("/data/train_final.csv")
df=df.fillna(0)

col=df.columns.tolist()
col.remove("id")
col.remove("Target")

x=df.loc[0:,col].values
y=df.loc[0:,"Target"].values

from keras.utils import np_utils

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3,random_state=42)

train_y_cat=np_utils.to_categorical(train_y,num_classes=None)
test_y_cat =np_utils.to_categorical(test_y,num_classes=None)

##scaling learned on train (i.e mean and sigma) and applied on train and test
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(train_x)
sc_train_x = scaler.transform(train_x)
sc_test_x = scaler.transform(test_x)

import keras
from keras.models import Sequential, Model
import keras
from keras.models import Sequential, Model 
from keras.layers import Dense,Input,Activation, Dropout

input_1 = Input(shape=sc_train_x.shape[1:]) 
x = Dense(50,activation="selu")(input_1)
x = Dropout(0.2)(x)
x = Dense(50,activation="selu")(x)
x = Dense(50,activation="selu")(x)
x = Dropout(0.2)(x)
x = Dense(50,activation="selu")(x)
x = Dense(50,activation="selu")(x)
x = Dense(2,activation="softmax")(x)
model = Model(input_1,x)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(sc_train_x,train_y_cat,epochs=10,batch_size=32,validation_data=(sc_test_x,test_y_cat))


y_pred = model.predict(sc_test_x)
y_pred_class = (y_pred >= 0.5)*1
from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, y_pred_class[0:,1:])
