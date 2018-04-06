# -*- coding: utf-8 -*-

# In[1]:


from gensim.models import word2vec
import gensim
import logging
import numpy as np
def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("D:\Topic\code\segment\compare\dataset\stanfordSentimentTreebank\standard\\train.txt")       #your data
    model = word2vec.Word2Vec(sentences, window=10, size=250,iter=10,min_count=32)
    model.save(u"English.model")
    print(model.similarity("movie","best"))

    #class gensim.models.word2vec.Word2Vec(sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)
  	

    # how to load a model ?
    # model = word2vec.Word2Vec.load_word2vec_format("your_model.bin", binary=True)
import gensim.models.keyedvectors as word2vec
if __name__ == '__main__':
    #main()
    model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #model = word2vec.Word2Vec.load("English.model")
    for e in model.most_similar(positive=['movie'], topn=10):
    	print(e[0], e[1])
    for e in model.most_similar(positive=['bad'], topn=10):
    	print(e[0], e[1])
    
    """
    w2v_model = gensim.models.KeyedVectors.load("category_7_3.model")
    print(w2v_model['美国'])
    print(np.asarray(w2v_model['美国'], dtype='float32'))
    """

