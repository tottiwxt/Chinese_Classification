# -*- coding: utf-8 -*-

# In[1]:


from gensim.models import word2vec
import logging

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("output_4_category.txt")       #your data
    model = word2vec.Word2Vec(sentences, window=6, size=250)
    model.save(u"category_4.model")
    print(model.similarity("刘翔","博尔特"))

    #class gensim.models.word2vec.Word2Vec(sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)
  	

    # how to load a model ?
    # model = word2vec.Word2Vec.load_word2vec_format("your_model.bin", binary=True)

if __name__ == '__main__':
    #main()
    model = word2vec.Word2Vec.load("category_4.model")
    for e in model.most_similar(positive=['学生'], topn=10):
    	print(e[0], e[1])
    for e in model.most_similar(positive=['股票'], topn=10):
    	print(e[0], e[1])
	

