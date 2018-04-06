
# coding: utf-8

x_data = []
y_data = []
from random import shuffle

rootdir = 'D:\Topic\code\segment\compare\dataset\stanfordSentimentTreebank\standard'

articles_train_text = open(rootdir + '/train.txt','r',encoding='utf-8').read().split('\n')
articles_train_label = open(rootdir + '/labels.txt','r',encoding='utf-8').read().split('\n')

articles_train = list(zip(articles_train_text, articles_train_label))


wordVectorLength = 250
#docVectorLength = 100
import gensim
import gensim.models.keyedvectors as word2vec
import numpy as np
categories = set(articles_train_label)
# Load word2vec model
w2v_model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#d2v_model = gensim.models.Doc2Vec.load('doc2vec.model')

articles_train_labels, articles_train_vectors = zip(*[
    (int(label), [w2v_model.wv[word]
      for word in text.split(' ') if word in w2v_model.wv])
      for text, label in articles_train
])

# articles_test_labels, articles_test_vectors = zip(*[
#     (int(label), [w2v_model.wv[word]
#       for word in text.split(' ') if word in w2v_model.wv])
#       for text, label in articles_test
# ])



articles_train_vectors = [[article[i] if len(article) > i else np.zeros(300) for i in range(20)] for article in articles_train_vectors]

#articles_test_vectors = [[article[i] if len(article) > i else np.zeros(wordVectorLength) for i in range(80)] for article in articles_test_vectors]

           
y_data_one_hot = np.zeros((len(articles_train_vectors), len(categories)))
y_data_one_hot[np.arange(len(articles_train_labels)), np.array(articles_train_labels)] = 1

x_data = articles_train_vectors
#print(articles_train_vectors[0])

# categories = set(articles_train_label)
# y_data_one_hot = np.zeros((len(y_data), len(categories)))
# y_data_one_hot[np.arange(len(y_data)), np.array(y_data)] = 1
# ### LSTM classification with keras LSTM cells

# In[73]:



# In[74]:

#size = len(x_data[0][0][0])
#print(x_data[0][0][0])
print(x_data[0][0])
data_dim = len(x_data[0][0])
timesteps = len(x_data[0])
#size = len(x_data[0][0][0])
num_classes = len(categories)

#print('size=',size);
print('data_dim=' ,data_dim);
print('timesteps=' , timesteps);

# In[75]:


split = 0.4
limit_train = (int)(len(x_data) * split)
# Generate dummy training data
x_train = x_data[:limit_train]
y_train = y_data_one_hot[:limit_train]

# Generate dummy validation data
x_val = x_data[limit_train:]
y_val = y_data_one_hot[limit_train:]




# print(y_train)




import keras
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
import numpy as np




# In[83]:
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import LSTM,Bidirectional
import numpy as np
from keras.layers.core import Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D,ConvLSTM2D,BatchNormalization

#input_shape = (timesteps, data_dim)
model = Sequential()


model.add(LSTM(30, return_sequences=True,dropout=0.5,
               input_shape=(timesteps, data_dim)))
            
model.add(Conv1D(32,
             3,
             padding='valid',
             activation='relu',
             strides=1
             #input_shape = ( timesteps, data_dim )
             ))
model.add(MaxPooling1D(pool_size=3))
#model.add(LSTM(50, return_sequences=True,dropout=0.5,
               #input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#model.add(Dropout(0.2))

#model.add(LSTM(50, dropout=0.5,return_sequences=True ,input_shape = ( timesteps, data_dim ,size)) ) # returns a sequence of vectors of dimension 32
#model.add(LSTM(50))  # return a single vector of dimension 32


model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_val, y_val, epochs=5, validation_data=(x_train, y_train))


# In[78]:


#print(model.evaluate(x_val, y_val))


# In[84]:


#model.predict([x_val[1]])


# In[ ]:


#y_val[1]


# In[ ]:




