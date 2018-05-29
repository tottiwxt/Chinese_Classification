
#import gensim_documents
import random
import gensim.parsing.preprocessing as preprocessing
import gensim.matutils
import gensim.models
import sklearn.metrics
import numpy as np

def process(doc):
  CUSTOM_FILTERS = [
    preprocessing.strip_tags,
    preprocessing.split_alphanum,
    preprocessing.strip_non_alphanum,
    preprocessing.strip_multiple_whitespaces,
    lambda d: d.lower(),
    lambda d: d.replace('!', '.'),
    lambda d: d.replace('?', '.')
  ]
  #print(' '.join(preprocessing.preprocess_string(doc, CUSTOM_FILTERS)))
  return ' '.join(preprocessing.preprocess_string(doc, CUSTOM_FILTERS))

def toVecs(content, model, word2vec=True):
  if word2vec:
    #base_wordvec = model.wv['足球']
    # for word in content.replace('.', ' ').split():
    #   if word in model.wv:
    #     wv = model.wv[word]
    #   else:
    #     wv = np.zeros(len(base_wordvec))
    #   for num in wv:
    #     yield num
    words = content.replace('.', ' ').split()
    vecs = [model.wv[words[word_index]]
            if word_index < len(words) and words[word_index] in model.wv
            else np.zeros(250)
            for word_index in range(100)]
    vecs = np.array(vecs).flatten()
    return vecs
  else:
    return gensim.matutils.unitvec(model.infer_vector(doc_words=content.split()))

def dot_product(doc1, doc2):
  return np.dot(doc1, doc2);

def main():
  # model = gensim.models.Doc2Vec.load("trained-sources/doc2vec_MM.model")

  rootdir = '/home/xutao/text_classification/data/six_categories/shuffle'

  articles_train_text = open(rootdir + '/train.txt','r',encoding='utf-8').read().split('\n')
  articles_train_label = open(rootdir + '/train_label2.txt','r',encoding='utf-8').read().split('\n')
  articles_train = list(zip(articles_train_text, articles_train_label))


  model = gensim.models.Word2Vec.load("/home/xutao/text_classification/data/w10iter10mini32_word2vec.model")
  #data = gensim_documents.MMDBDocumentLists('../MM/csv_by_category/', useHeading=True, limit=-1)
  processed_data = [(toVecs(process(text), model), label) for text, label in articles_train]
  docs, categories = zip(*processed_data)
  print(docs[0].shape)
  print(sklearn.metrics.silhouette_score(docs, categories))


if __name__ == '__main__':
  main()
