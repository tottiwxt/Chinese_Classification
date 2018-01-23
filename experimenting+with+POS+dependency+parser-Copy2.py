
# coding: utf-8

# In[1]:
"""
def loadLibFolder (folder):
    import os, sys
    if folder not in sys.path:
        sys.path.insert(1, os.path.join(sys.path[0], folder))


# # Experimenting with POS dependency parser
# To be able to predict a category out of a sentence/text it is assumed that the POS tags and the dependency tree could have an inpact on the result. Here we investigate that relation

# In[2]:


from urllib import request, parse
import json
url = 'http://localhost:1337/sentence/'


# ## Sample text to try out the parser

# In[3]:


def parseSentence(sentence):
    try:
        sentence = request.quote(sentence)
        f =  request.urlopen(url + sentence)
        res = json.loads(f.read().decode('latin1'))
        return res
    except:
        return {'sentenceData': []}
def onlyNounsAndVerbs(data):
    return {
        'sentenceData': [word for word in data['sentenceData'] if 'NN' in word['tag'].split('|') or 'VB' in word['tag'].split('|')]
    }
def untilLevel(level, data):
    return {
        'sentenceData': [word for word in data['sentenceData'] if (int)(word['parent']) <= level]
    }
def toWordArray(data):
    return [word['base_word'] for word in data['sentenceData']]


# In[4]:


res = parseSentence('Han ler mot henne och hela hans ansikte säger att han älskar henne med hela sitt hjärta')


# In[5]:


# Example filtering
print ("Raw data:")
print (res)
print ("All words:")
print ([word['word'] for word in res['sentenceData']])
print ("Level three data:")
print ([word['word']+ '::' + word['tag'].split('|')[0] for word in res['sentenceData'] if (int)(word['parent']) <= 3])
print ("Only nouns and verbs:")
print ([word['word'] for word in res['sentenceData'] if 'NN' in word['tag'].split('|') or 'VB' in word['tag'].split('|')])

print(" ".join(toWordArray(untilLevel(3, onlyNounsAndVerbs(parseSentence(res))))))


# ## Classification experiment

# In[7]:


loadLibFolder('../gensim')

import os
import gensim
import gensim_documents
import dotenv
import numpy as np
dotenv.load()


# In[ ]:


limit_per_category = 2000
use_cache = False
use_all_data = True


# In[ ]:


categories = []
x_data = []
y_data = []
model = gensim.models.Doc2Vec.load(dotenv.get('DOC2VEC_MODEL'))

if use_cache and os.path.isfile('data/tmp_dependency_data'):
    with open('data/tmp_dependency_data', 'r', encoding='utf-8', errors='ignore') as tmp_cache_file:
        for category in tmp_cache_file:
            category = category[:-1]
            if category == "\n": continue
            if category not in categories:
                print ("TT", category)
                categories.append(category)
            sentVecs = []
            while True:
                sentence = tmp_cache_file.readline()[:-1]
                if sentence == "":
                    break
                artvec = model.infer_vector(doc_words=sentence.split())
                sentVecs.append(gensim.matutils.unitvec(artvec))
            y_data.append(categories.index(category))
            x_data.append(sentVecs)
else:
    data = gensim_documents.MMDBDocumentLists(dotenv.get('ARTICLE_PATH', '.') + '/csv_by_category/', useHeading=True, limit=limit_per_category)
    # with open('data/tmp_dependency_data', 'w', encoding='utf-8', errors='ignore') as tmp_cache_file:
    for i, doc in enumerate(data):
        if not doc.category in categories:
            categories.append(doc.category)
        #tmp_cache_file.write(doc.category + "\n")

        sentences = doc.content.split(".")
        sentVecs = []
        for j in range(20):
            if j >= len(sentences): 
                sentVecs.append(np.zeros(300))
                continue
            if use_all_data:
                sentence = sentences[j]
            else:
                sentence = " ".join(toWordArray(untilLevel(3, onlyNounsAndVerbs(parseSentence(sentences[j])))))
            if sentence == "":
                sentVecs.append(np.zeros(300))
                continue
            artvec = model.infer_vector(doc_words=sentence.split())
            sentVecs.append(gensim.matutils.unitvec(artvec))
            #tmp_cache_file.write(sentence + "\n")
        #tmp_cache_file.write("\n")
        x_data.append(sentVecs)
        y_data.append(categories.index(doc.category))

        if i % (limit_per_category/4) == 0:
            print ("New epoch started, nr.", i+1, " of ", len(categories) * limit_per_category, " epochs")


# In[ ]:



"""

# #### Encode one hot vectors for the classes

# In[72]:
x_data = []
y_data = []
from random import shuffle
articles_train_text = open('segment/data/lstm_train/doc2Vec/train.txt','r',encoding='utf-8').read().split('\n')
articles_train_label = open('segment/data/lstm_train/doc2Vec/train_label2.txt','r',encoding='utf-8').read().split('\n')
articles_train = list(zip(articles_train_text, articles_train_label))
shuffle(articles_train)
articles_test_text = open('segment/data/lstm_train/doc2Vec/test.txt','r',encoding='utf-8').read().split('\n')
articles_test_label = open('segment/data/lstm_train/doc2Vec/test_label2.txt','r',encoding='utf-8').read().split('\n')
articles_test = list(zip(articles_test_text, articles_test_label))
shuffle(articles_test)

wordVectorLength = 250
docVectorLength = 100
import gensim
import numpy as np
categories = set(articles_train_label)
# Load word2vec model
#w2v_model = gensim.models.Word2Vec.load('segment/category_7_3.model')
d2v_model = gensim.models.Doc2Vec.load('segment/doc2vec.model')
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
x_train = x_data[:limit_train]
y_train = y_data_one_hot[:limit_train]

# Generate dummy validation data
x_val = x_data[limit_train:]
y_val = y_data_one_hot[limit_train:]

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
model = Sequential()
model.add(LSTM(50, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(50))  # return a single vector of dimension 32
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
#model.add(Dense(10, activation='softmax'))
#model.add(Flatten())
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))


# In[78]:


model.evaluate(x_val, y_val)


# In[84]:


model.predict([x_val[1]])


# In[ ]:


y_val[1]


# In[ ]:




