# Read the data

import numpy as np
import tensorflow as tf
import random as rn

import os
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


import pandas as pd
df = pd.read_csv('/data/kiosk_data_final_v1.csv')

# without merge
y = df['y'].values

col = df.columns.tolist()
col.remove('y')
col.remove('id')
x = df[col].values

# Split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)

#Model Development
from keras.layers.embeddings import Embedding
from keras.layers import Dense,Input,Activation,Flatten
from keras.models import Sequential, Model

inputs = Input(shape=(x.shape[1],))
x = Dense(10,activation="relu")(inputs)
x = Dense(10,activation="relu")(x)
x = Dense(1)(x)
model = Model(inputs,x)
model.compile(loss="mean_squared_error",optimizer="adam")
model.fit(x_train,y_train,epochs=10,batch_size=10,validation_data=(x_test,y_test))

#with Merge
# Read the data
import numpy as np
import tensorflow as tf
import random as rn

import os
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


import pandas as pd
df = pd.read_csv('/data/dl/kiosk_data_final_v1.csv')

# without merge
y = df['y'].values

col = df.columns.tolist()
col.remove('y')
col.remove('id')
col.remove('day_open_per')
col.remove('night_open_per')
x1 = df[col].values
x2 = df[['night_open_per','day_open_per']].values

# Split the data
from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y_train,y_test = train_test_split(x1,x2,y,test_size=0.30)

#Model Development
from keras.layers.embeddings import Embedding
from keras.layers import Dense,Input,Activation,Flatten
from keras.models import Sequential, Model
from keras.layers import Add, multiply,Concatenate
import keras


input1 = Input(shape=(x1.shape[1],))
x = Dense(10,activation="relu")(input1)
x = Dense(10,activation="relu")(x)
x = Dense(1)(x)

input2 = Input(shape=(x2.shape[1],))
x1 = Dense(1, activation='sigmoid')(input2)

out_all  = keras.layers.multiply([x,x1])
out = Dense(1)(out_all)

model = Model(inputs=[input1,input2], outputs=out)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit([x1_train,x2_train],y_train,validation_data=([x1_test,x2_test],y_test),epochs=10,batch_size=10)