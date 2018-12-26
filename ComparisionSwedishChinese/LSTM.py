# coding: utf-8



x_data = []
y_data = []
from random import shuffle
articles_train_text = open('data/train.txt','r',encoding='utf-8').read().split('\n')
articles_train_label = open('data/label.txt','r',encoding='utf-8').read().split('\n')
articles_train = list(zip(articles_train_text, articles_train_label))
shuffle(articles_train)


wordVectorLength = 250
docVectorLength = 100
import gensim
import numpy as np
categories = set(articles_train_label)
# Load word2vec model
w2v_model = gensim.models.Word2Vec.load('../data/w10iter10mini32_word2vec.model')
#d2v_model = gensim.models.Doc2Vec.load('segment/doc2vec.model')

articles_train_labels, articles_train_vectors = zip(*[
    (int(label), [w2v_model.wv[word]
      for word in text.split(' ') if word in w2v_model.wv])
      for text, label in articles_train
])


import re
# for text , label in articles_train:
#     #sentences = re.split(r'。\?？!！',text)
#     sentences = text.split('。')
#     sentVecs = []
#     #print(len(sentences))
#     for j in range(20):
#         if j >= len(sentences):
#             sentVecs.append(np.zeros(100))
#             continue
#         else:
#             sentence =  re.sub('。？?！!',' ',sentences[j].strip())
#         if sentence == "":
#             sentVecs.append(np.zeros(100))
#             continue
#         artvec = d2v_model.infer_vector(doc_words=sentence.split())
#         sentVecs.append(gensim.matutils.unitvec(artvec))
#     x_data.append(sentVecs)
#     y_data.append(int(label)) 

# for text , label in articles_test:
#     sentences = text.split('。')
#     sentVecs = []
#     for j in range(20):
#         if j >= len(sentences):
#             sentVecs.append(np.zeros(100))
#             continue
#         else:
#             sentence =  re.sub('。？?！!',' ',sentences[j].strip())
#         if sentence == "":
#             sentVecs.append(np.zeros(100))
#             continue
#         artvec = d2v_model.infer_vector(doc_words=sentence.split())
#         sentVecs.append(gensim.matutils.unitvec(artvec))
#     x_data.append(sentVecs)
#     y_data.append(int(label))

###

articles_train_vectors = [[article[i] if len(article) > i else np.zeros(wordVectorLength) for i in range(80)] for article in articles_train_vectors]
# articles_test_vectors = [[article[i] if len(article) > i else np.zeros(wordVectorLength) for i in range(30)] for article in articles_train_vectors]



y_data_one_hot = np.zeros((len(articles_train_vectors), len(categories)))
y_data_one_hot[np.arange(len(articles_train_labels)), np.array(articles_train_labels)] = 1

x_data = articles_train_vectors
articles_train_vectors[0]

# categories = set(articles_train_label)
# y_data_one_hot = np.zeros((len(y_data), len(categories)))
# y_data_one_hot[np.arange(len(y_data)), np.array(y_data)] = 1
# ### LSTM classification with keras LSTM cells

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np




data_dim = len(x_data[0][0])
timesteps = len(x_data[0])
num_classes = len(categories)


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
    
print(len(x_val))
print(len(x_val[0]))
print(len(x_val[0][0]))


print(y_train)


x_train = np.array(x_train)

y_train = np.array(y_train)

x_val = np.array(x_val)

y_val = np.array(y_val)

x_test = np.array(x_test)

y_test = np.array(y_test)
n_layers = 10 
num_classes = 14 
model = Sequential()
model.add(LSTM(50,
        input_shape=(timesteps, data_dim),
        return_sequences=True))  # returns a sequence of vectors of dimension 50
for layer in range(n_layers-2):
      model.add(Dropout(0.2))
      model.add(LSTM(50, return_sequences=True))  # returns a sequence of vectors of dimension 50
#model.add(LSTM(50))  # return a single vector of dimension 50


model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))



model.evaluate(x_test, y_test)



prediction = model.predict(np.array(x_val))
copy_prediction = prediction
copy_prediction = [[1.0 if max(y) == i else 0.0 for i in y] for y in prediction]   

from sklearn.metrics import confusion_matrix
import pandas
categories = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
conf_mat = confusion_matrix([categories[y.argmax()] for y in y_val], [categories[y.argmax()] for y in np.array(copy_prediction)])
print(pandas.DataFrame(conf_mat, columns=categories, index=categories))




