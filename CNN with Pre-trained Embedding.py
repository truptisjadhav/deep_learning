import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
import os
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D


# Glove data 
f = open(os.path.join('/data/text/', 'glove.6B.50d.txt'))
cnt =0
for i in f:
    a=i.split()
    print(a)
    cnt = cnt+1
    if cnt == 4: break
f.close()

# Load the data into key value pair - word and embeddings
f = open(os.path.join('/data/text/', 'glove.6B.50d.txt'))
embedding_dict={}

for i in f:
    emd     =i.split()
    word    =emd[0]
    emd_vec =emd[1:]
    embedding_dict[word]=emd_vec

f.close()
#first2pairs = {k: embedding_dict[k] for k in list(embedding_dict)[:2]}

#import Data
X = np.load("/data/text/text_x_data.npy")
Y = np.load("/data/text/text_y_data.npy")

#Create the tokens and vocab dictionary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
word_count = len(tokenizer.word_counts)
word_index = tokenizer.word_index

#max length
maxlen=max([len(i) for i in X_seq])
X_pad =pad_sequences(X_seq, maxlen=maxlen)

#One hot encoding Y
#See number of categories collections.Counter(Y)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y_int = label_encoder.fit_transform(Y)
Y_cat =to_categorical(Y_int)


#splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_pad,Y_cat,test_size=0.30)

#Create embedding layer
from keras.layers.embeddings import Embedding
from keras.layers import Dense,Input,Activation,Flatten,LSTM
from keras.models import Sequential, Model


# Creating the matrix for word_index and embedding. Initialize to zero
# this becomes the kind of lookup values
# embedding_dim is the size of embeddings

embedding_dim = 50
embedding_matrix = np.zeros((word_count+ 1, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(word_count+ 1,embedding_dim,input_length=maxlen,trainable=False,weights=[embedding_matrix])

sequence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Convolution1D(40,5,activation='relu',strides=1)(embedded_sequences)
x = Flatten()(x)
x = Dense(2,activation="softmax")(x)
model = Model(sequence_input,x)

# look at model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=500,validation_data=(x_test,y_test))

# predict 
predict = model.predict(x_test)