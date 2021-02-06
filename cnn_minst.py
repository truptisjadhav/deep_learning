import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
print(y_train[0])

y_cat = keras.utils.to_categorical(y_train, num_classes=None)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=None)
x_train1 = x_train.reshape(x_train.shape[0] ,28, 28,1).astype('float32')
x_test1 = x_test.reshape(x_test.shape[0] ,28, 28,1).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train1 = x_train1 / 255
x_test1 = x_test1 / 255


input1=Input(shape=(28,28,1,))
x1 = Conv2D(32, kernel_size=(5, 5),activation='relu')(input1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Flatten()(x1)
x1 = Dense(128, activation='relu') (x1)
x1 = Dense(10, activation='softmax')(x1)
model = Model(inputs=[input1], outputs=x1)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train1,y_cat,batch_size=32,epochs=10,validation_data=(x_test1,y_test_cat))  


###another
input1=Input(shape=(28,28,1,))
x1 = Conv2D(32, kernel_size=(5, 5),activation='relu')(input1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Conv2D(16, kernel_size=(3, 3),activation='relu')(input1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Flatten()(x1)
x1 = Dense(128, activation='relu') (x1)
x1 = Dense(10, activation='softmax')(x1)
model = Model(inputs=[input1], outputs=x1)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train1,y_cat,batch_size=32,epochs=10,validation_data=(x_test1,y_test_cat))  

# save model
# model.save('/data/video/common/minst_model.h5')