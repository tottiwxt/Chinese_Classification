
# coding: utf-8


# #### Encode one hot vectors for the classes

# In[72]:
x_data = []
y_data = []
from random import shuffle
articles_train_text = open('d:/Topic/code/segment/data/lstm_train/doc2Vec/train.txt','r',encoding='utf-8').read().split('\n')
articles_train_label = open('d:/Topic/code/segment/data/lstm_train/doc2Vec/train_label2.txt','r',encoding='utf-8').read().split('\n')
articles_train = list(zip(articles_train_text, articles_train_label))
shuffle(articles_train)
articles_test_text = open('d:/Topic/code/segment/data/lstm_train/doc2Vec/test.txt','r',encoding='utf-8').read().split('\n')
articles_test_label = open('d:/Topic/code/segment/data/lstm_train/doc2Vec/test_label2.txt','r',encoding='utf-8').read().split('\n')
articles_test = list(zip(articles_test_text, articles_test_label))
shuffle(articles_test)

wordVectorLength = 250
docVectorLength = 100
import gensim
import numpy as np
categories = set(articles_train_label)
# Load word2vec model
#w2v_model = gensim.models.Word2Vec.load('segment/category_7_3.model')
d2v_model = gensim.models.Doc2Vec.load('d:/Topic/code/segment/doc2vec.model')
"""
articles_train_labels, articles_train_vectors = zip(*[
    (int(label), [w2v_model.wv[word]
      for word in text.split(' ') if word in w2v_model.wv])
      for text, label in articles_train
])
articles_train_labels, articles_test_vectors = zip(*[
    (int(label), [w2v_model.wv[word]
      for word in text.split(' ') if word in w2v_model.wv])
      for text, label in articles_test
])
"""
import re
for text , label in articles_train:
    #sentences = re.split(r'。\?？!！',text)
    sentences = text.split('。')
    sentVecs = []
    #print(len(sentences))
    for j in range(20):
        if j >= len(sentences):
            sentVecs.append(np.zeros(100))
            continue
        else:
            sentence =  re.sub('。？?！!',' ',sentences[j].strip())
        if sentence == "":
            sentVecs.append(np.zeros(100))
            continue
        artvec = d2v_model.infer_vector(doc_words=sentence.split())
        sentVecs.append(gensim.matutils.unitvec(artvec))
    x_data.append(sentVecs)
    y_data.append(int(label)) 

for text , label in articles_test:
    sentences = text.split('。')
    sentVecs = []
    for j in range(20):
        if j >= len(sentences):
            sentVecs.append(np.zeros(100))
            continue
        else:
            sentence =  re.sub('。？?！!',' ',sentences[j].strip())
        if sentence == "":
            sentVecs.append(np.zeros(100))
            continue
        artvec = d2v_model.infer_vector(doc_words=sentence.split())
        sentVecs.append(gensim.matutils.unitvec(artvec))
    x_data.append(sentVecs)
    y_data.append(int(label))



"""
print ('out: ')
print(len([word for article in articles_train_text for word in article.split(' ') if not word in w2v_model.wv]))
print ('in')
print(len([word for article in articles_train_text for word in article.split(' ') if  word in w2v_model.wv]))

articles_train_vectors = [[article[i] if len(article) > i else np.zeros(wordVectorLength) for i in range(30)] for article in articles_train_vectors]
articles_test_vectors = [[article[i] if len(article) > i else np.zeros(wordVectorLength) for i in range(30)] for article in articles_train_vectors]
"""

"""
y_data_one_hot = np.zeros((len(articles_train_vectors), len(categories)))
y_data_one_hot[np.arange(len(articles_train_labels)), np.array(articles_train_labels)] = 1

x_data = articles_train_vectors
articles_train_vectors[0]
"""
categories = set(articles_train_label)
y_data_one_hot = np.zeros((len(y_data), len(categories)))
y_data_one_hot[np.arange(len(y_data)), np.array(y_data)] = 1
# ### LSTM classification with keras LSTM cells

# In[73]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np


# In[74]:


data_dim = len(x_data[0][0])
timesteps = len(x_data[0])
num_classes = len(categories)


# In[75]:


split = 0.4
limit_train = (int)(len(x_data) * split)
# Generate dummy training data
x_train = x_data[:9800]
y_train = y_data_one_hot[:9800]

print('shape of y:' )
print( y_train[0])

# Generate dummy validation data
x_val = x_data[9801:14000]
y_val = y_data_one_hot[9801:14000]

print('matrix = 0 :')
# In[82]:
print(len([word for article in x_train for word in article if sum(word) == 0]))
print('matrix != 0 :')
# In[82]:
print(len([word for article in x_train for word in article if sum(word) != 0]))

print(len(x_val))
print(len(x_val[0]))
print(len(x_val[0][0]))

print(len(x_train))
print(len(x_train[0]))
print(len(x_train[0][0]))
print(len(categories))

print(y_train)


# To train a Sequential LSTM model that can classify a stacked sequence of words we need to define the input as follows:
#  * batch_size - number of datapoints in the dataset
#  * timesteps - the number of words per sequence
#  * data_dim - the number of features per word instance

# In[83]:


# expected input data shape: (batch_size, timesteps, data_dim)

# expected input data shape: (batch_size, timesteps, data_dim)
input_shape = (timesteps, data_dim)
model = Sequential()
model.add(LSTM(50, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
for layer in range(8):
    model.add(LSTM(50, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(50))  # return a single vector of dimension 32
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# In[78]:
    




print ('(6) testing model...')
print (model.evaluate(x_val, y_val))
#print(x_val[0])
prediction = model.predict(np.array(x_val))
copy_prediction = prediction
copy_prediction = [[1.0 if max(y) == i else 0.0 for i in y] for y in prediction]   
    
from sklearn.metrics import confusion_matrix
import pandas
categories = [0,1,2,3,4,5,6]
conf_mat = confusion_matrix([categories[y.argmax()] for y in y_val], [categories[y.argmax()] for y in np.array(copy_prediction)])
print(pandas.DataFrame(conf_mat, columns=categories, index=categories))

print("END")


