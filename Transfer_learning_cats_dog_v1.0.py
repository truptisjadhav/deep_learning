from os import listdir
from sklearn.model_selection import train_test_split
import cv2
import os
from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras.utils import np_utils
import keras
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# Data needs to be centered

path = '/data/video/cat_dog/train'
file_list =listdir(path)

train, other =train_test_split(file_list,train_size=0.02) 
test, discard =train_test_split(other,train_size=0.01)

train_x =[]
train_y =[1.0 if 'dog' in i else 0.0 for i in train] 
train_y_cat = keras.utils.to_categorical(train_y, num_classes=None)

for i in train:
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype('float32'))
    train_x.append(img)

train_x=np.array(train_x)
train_x.shape

test_x =[]
test_y =[1.0 if 'dog' in i else 0.0 for i in test] 
test_y_cat = keras.utils.to_categorical(test_y, num_classes=None)

for i in test:
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype('float32'))
    test_x.append(img)

test_x=np.array(test_x)
test_x.shape

base_model = applications.VGG16(weights='imagenet', include_top=False,input_shape=(224,224,3))

new_model = Sequential()

for layers in base_model.layers:
    layers.trainable = False
    new_model.add(layers)


new_model.add(Flatten())
new_model.add(Dense(2, activation='softmax'))

new_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
new_model.fit(train_x_pred,train_y_cat,batch_size=32,epochs=10,validation_data=(test_x_pred,test_y_cat))