
# coding: utf-8

x_data = []
y_data = []
from random import shuffle

#中文数据读入
rootdir = '/Users/wangxutao/Programming/Chinese_Classification/data/ALL'

articles_train_text = open(rootdir + '/train.txt','r',encoding='utf-8').read().split('\n')
articles_train_label = open(rootdir + '/label.txt','r',encoding='utf-8').read().split('\n')
# articles_test_text = open(rootdir + '/test.txt','r',encoding='utf-8').read().split('\n')
# articles_test_label = open(rootdir + '/test_label2.txt','r',encoding='utf-8').read().split('\n')
articles_train = list(zip(articles_train_text, articles_train_label))
# articles_test = list(zip(articles_test_text, articles_test_label))
shuffle(articles_train)
'''
#英文数据读入
rootdir ='/Users/wangxutao/Programming/Chinese_Classification/data/English/shuffle'
articles_train_text = open(rootdir + '/train.txt' , 'r', encoding='utf-8').read().split('\n')
articles_train_label = open(rootdir + '/label.txt' , 'r', encoding='utf-8').read().split('\n')
articles_train = list(zip(articles_train_text,articles_train_label))
# shuffle(articles_train)
'''
wordVectorLength = 250
# 英文词向量维度
# wordVectorLength = 300
docVectorLength = 100
import gensim
import numpy as np
categories = set(articles_train_label)
# Load word2vec model
# 中文词向量
w2v_model = gensim.models.Word2Vec.load('/Users/wangxutao/Programming/Chinese_Classification/data/w10iter10mini32_word2vec.model')
# 英文词向量

# w2v_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/wangxutao/Programming/Chinese_Classification/data/GoogleNews-vectors-negative300.bin', binary=True)
#d2v_model = gensim.models.Doc2Vec.load('doc2vec.model')

articles_train_labels, articles_train_vectors = zip(*[
    (int(label), [w2v_model.wv[word]
      for word in text.split(' ') if word in w2v_model.wv])
      for text, label in articles_train
])
#print(articles_train_vectors[0])

# articles_test_labels, articles_test_vectors = zip(*[
#     (int(label), [w2v_model.wv[word]
#       for word in text.split(' ') if word in w2v_model.wv])
#       for text, label in articles_test
# ])


article_length = 80
articles_train_vectors = [[article[i] if len(article) > i else np.zeros(wordVectorLength) for i in range(article_length)] for article in articles_train_vectors]


           
y_data_one_hot = np.zeros((len(articles_train_vectors), len(categories)))
y_data_one_hot[np.arange(len(articles_train_labels)), np.array(articles_train_labels)] = 1

x_data = articles_train_vectors

# # x_test = articles_test_vectors
# y_test_one_hot = np.zeros((len(articles_test_vectors),len(categories)))
# y_test_one_hot[np.arange(len(articles_test_labels)), np.array(articles_test_labels)] = 1

print(x_data[0][0])
data_dim = len(x_data[0][0])
timesteps = len(x_data[0])

num_classes = len(categories)
print('num_classes = ' + str(num_classes))

print('data_dim=' ,data_dim);
print('timesteps=' , timesteps);


split = 0.6
split2 = 0.8
limit_train = (int)(len(x_data) * split)
limit_train2 = (int)(len(x_data) * split2)
# Generate dummy training data
x_train = x_data[:limit_train]
y_train = y_data_one_hot[:limit_train]

# Generate dummy validation data
x_val = x_data[limit_train:limit_train2]
y_val = y_data_one_hot[limit_train:limit_train2]

x_test = x_data[limit_train2:]
y_test = y_data_one_hot[limit_train2:]


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

# model.add(LSTM(50, return_sequences=True,dropout=0.5,
#                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.2))


model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), epochs=10, validation_data=(np.array(x_test), np.array(y_test)))
# print(model.evaluate(np.array(x_val),np.array(y_val)))


prediction = model.predict(np.array(x_val))
copy_prediction = prediction
copy_prediction = [[1.0 if max(y) == i else 0.0 for i in y] for y in prediction]   
    
from sklearn.metrics import confusion_matrix
import pandas
categories = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
conf_mat = confusion_matrix([categories[y.argmax()] for y in y_val], [categories[y.argmax()] for y in np.array(copy_prediction)])
print(pandas.DataFrame(conf_mat, columns=categories, index=categories))




