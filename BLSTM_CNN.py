
# coding: utf-8

x_data = []
y_data = []
from random import shuffle

rootdir = 'D:/Topic/code/segment/data/lstm_train/7_Confuse/shuffle'

articles_train_text = open(rootdir + '/train.txt','r',encoding='utf-8').read().split('\n')
articles_train_label = open(rootdir + '/train_label2.txt','r',encoding='utf-8').read().split('\n')
articles_test_text = open(rootdir + '/test.txt','r',encoding='utf-8').read().split('\n')
articles_test_label = open(rootdir + '/test_label2.txt','r',encoding='utf-8').read().split('\n')
articles_train = list(zip(articles_train_text, articles_train_label))
articles_test = list(zip(articles_test_text, articles_test_label))

wordVectorLength = 250
docVectorLength = 100
import gensim
import numpy as np
categories = set(articles_train_label)
# Load word2vec model
w2v_model = gensim.models.Word2Vec.load('D:/Topic/code/segment/w10iter10mini32_word2vec.model')
#d2v_model = gensim.models.Doc2Vec.load('doc2vec.model')

articles_train_labels, articles_train_vectors = zip(*[
    (int(label), [w2v_model.wv[word]
      for word in text.split(' ') if word in w2v_model.wv])
      for text, label in articles_train
])
# articles_train_labels, articles_train_vectors = zip(*[
#     (int(label), [[w2v_model.wv[word]
#       for word in sentence.split(' ') if word in w2v_model.wv]
#       for sentence in sentences.split('。')])
#       for sentences, label in articles_train
      
# ])
#print(articles_train_vectors[0])

# articles_test_labels, articles_test_vectors = zip(*[
#     (int(label), [w2v_model.wv[word]
#       for word in text.split(' ') if word in w2v_model.wv])
#       for text, label in articles_test
# ])
# articles_test_labels, articles_test_vectors = zip(*[
#     (int(label), [[w2v_model.wv[word]
#       for word in sentence.split(' ') if word in w2v_model.wv]
#       for sentence in sentences.split('。')])
#       for sentences, label in articles_test
      
# ])


article_length = 80
articles_train_vectors = [[article[i] if len(article) > i else np.zeros(wordVectorLength) for i in range(article_length)] for article in articles_train_vectors]


           
y_data_one_hot = np.zeros((len(articles_train_vectors), len(categories)))
y_data_one_hot[np.arange(len(articles_train_labels)), np.array(articles_train_labels)] = 1

x_data = articles_train_vectors

x_test = articles_test_vectors
y_test_one_hot = np.zeros((len(articles_test_vectors),len(categories)))
y_test_one_hot[np.arange(len(articles_test_labels)), np.array(articles_test_labels)] = 1

print(x_data[0][0])
data_dim = len(x_data[0][0])
timesteps = len(x_data[0])

num_classes = len(categories)


print('data_dim=' ,data_dim);
print('timesteps=' , timesteps);



split = 0.2
limit_train = (int)(len(x_data) * split)
# Generate dummy training data
x_train = x_data[:limit_train]
y_train = y_data_one_hot[:limit_train]

# Generate dummy validation data
x_val = x_data[limit_train:]
y_val = y_data_one_hot[limit_train:]


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import LSTM,Bidirectional
import numpy as np
from keras.layers.core import Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D,ConvLSTM2D,BatchNormalization

input_shape = (timesteps, data_dim)

model = Sequential()

model.add(Bidirectional(LSTM(50, return_sequences=True,dropout=0.5),
               input_shape=(timesteps, data_dim)))
model.add(Conv1D(64,
             7,
             padding='valid',
             activation='relu',
             strides=1
             #input_shape = ( timesteps, data_dim )
             ))
model.add(MaxPooling1D(pool_size=4))

model.add(Dropout(0.2))


model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

'''
#10-crossvalidation
n_folds = 10
labels = to_categorical(np.asarray(articles_train_label))
c, r = labels.shape
labels = y_data.reshape(c,)

skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D
from sklearn.cross_validation import StratifiedKFold
for i, (train, test) in enumerate(skf):
    print ("Running Fold", i+1, "/", n_folds)
    model = Sequential()

    model.add(Bidirectional(LSTM(50, return_sequences=True,dropout=0.5),
                   input_shape=(timesteps, data_dim)))
    model.add(Conv1D(64,
                 5,
                 padding='valid',
                 activation='relu',
                 strides=1
                 #input_shape = ( timesteps, data_dim )
                 ))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Dropout(0.2))


    model.add(Flatten())

    model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    # print('train='+ data[train])
    # print('lable='+ labels[train])
    
    labels1 = to_categorical(labels[train])
    labels2 = to_categorical(labels[test])
    model.fit(data[train], labels1, epochs=1, batch_size=128)
    print(model.evaluate(data[test], labels2))

'''
#10-

#print(model.evaluate(x_test, y_test))








