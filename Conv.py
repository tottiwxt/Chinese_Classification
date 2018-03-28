
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
w2v_model = gensim.models.Word2Vec.load('w10iter10mini32_word2vec.model')
d2v_model = gensim.models.Doc2Vec.load('doc2vec.model')

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

articles_test_labels, articles_test_vectors = zip(*[
    (int(label), [w2v_model.wv[word]
      for word in text.split(' ') if word in w2v_model.wv])
      for text, label in articles_test
])
# articles_test_labels, articles_test_vectors = zip(*[
#     (int(label), [[w2v_model.wv[word]
#       for word in sentence.split(' ') if word in w2v_model.wv]
#       for sentence in sentences.split('。')])
#       for sentences, label in articles_test
      
# ])



articles_train_vectors = [[article[i] if len(article) > i else np.zeros(wordVectorLength) for i in range(80)] for article in articles_train_vectors]

articles_test_vectors = [[article[i] if len(article) > i else np.zeros(wordVectorLength) for i in range(80)] for article in articles_test_vectors]
#articles_train_vectors =[[[sentence[i] if len(sentence) > i else np.zeros(wordVectorLength) for i in range(20)]for sentence in article] for article in articles_train_vectors]

#articles_test_vectors =[[[article[j][i] if len(article[j]) > i else np.zeros(wordVectorLength) for i in range(20)] for j in range(5) ] for article in articles_test_vectors]
#articles_train_vectors =[[[article[j][i] if len(article[j]) > i else np.zeros(wordVectorLength) for i in range(20)] for j in range(5) ] for article in articles_train_vectors]
#articles_train_vectors = [ [ [([article[j][i] if len(article[j]) > i else np.zeros(wordVectorLength) for i in range(20)]) if len(article)>j  else ([np.zeros(wordVectorLength) for i in range(20)])] for j in range(5)]  for article in articles_train_vectors  ]
#articles_train_vectors =[[[article[j][i] if len(article)>j and len(article[j]) > i  else np.zeros(wordVectorLength) for i in range(20) ] for j in range(5) ] for article in articles_train_vectors]

           
y_data_one_hot = np.zeros((len(articles_train_vectors), len(categories)))
y_data_one_hot[np.arange(len(articles_train_labels)), np.array(articles_train_labels)] = 1

x_data = articles_train_vectors
print("---------------------------sdtsrt--------------------------" )
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


split = 0.2
limit_train = (int)(len(x_data) * split)
# Generate dummy training data
x_train = x_data[:limit_train]
y_train = y_data_one_hot[:limit_train]

# Generate dummy validation data
x_val = x_data[limit_train:]
y_val = y_data_one_hot[limit_train:]

# print('matrix = 0 :')
# # In[82]:
# print(len([word for article in x_train for word in article if sum(word) == 0]))
# print('matrix != 0 :')
# # In[82]:
# print(len([word for article in x_train for word in article if sum(word) != 0]))

# print(len(x_val))
# print(len(x_val[0]))
# print(len(x_val[0][0]))

# print(len(x_train))
# print(len(x_train[0]))
# print(len(x_train[0][0]))
# print(len(categories))

# print(y_train)


# To train a Sequential LSTM model that can classify a stacked sequence of words we need to define the input as follows:
#  * batch_size - number of datapoints in the dataset
#  * timesteps - the number of words per sequence
#  * data_dim - the number of features per word instance

# In[83]:
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import LSTM,Bidirectional
import numpy as np
from keras.layers.core import Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D,ConvLSTM2D,BatchNormalization
# expected input data shape: (batch_size, timesteps, data_dim)
input_shape = (timesteps, data_dim)
model = Sequential()
# model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
#                      activation='tanh',
#                      return_sequences=True,
#                      padding='same',
#                      input_shape=(None      ,timesteps, data_dim ,size),
#                      name='FirstCLSTMv2'))
# model.add(BatchNormalization())
#model.add(Reshape(input_shape +(1, ),input_shape=input_shape))
model.add(LSTM(50, return_sequences=True,dropout=0.5,
               input_shape=(timesteps, data_dim)))
model.add(Conv1D(128,
             6,
             padding='valid',
             activation='relu',
             strides=1
             #input_shape = ( timesteps, data_dim )
             ))
model.add(MaxPooling1D(pool_size=4))
#model.add(LSTM(50, return_sequences=True,dropout=0.5,
               #input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.2))

#model.add(LSTM(50, dropout=0.5,return_sequences=True ,input_shape = ( timesteps, data_dim ,size)) ) # returns a sequence of vectors of dimension 32
#model.add(LSTM(50))  # return a single vector of dimension 32
#model.add(ConvLSTM2D(64, (3, 3), activation='tanh', name='secondLSTM'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))


# In[78]:


print(model.evaluate(x_val, y_val))


# In[84]:


#model.predict([x_val[1]])


# In[ ]:


#y_val[1]


# In[ ]:




