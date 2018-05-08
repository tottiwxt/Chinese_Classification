# -*- coding: utf-8 -*-
# word_segment.py用于语料分词

import logging
import os.path
import sys
import re
import jieba


# 先用正则将<content>和</content>去掉
def reTest(content):
  reContent = re.sub('<content>|</content>','',content)
  return reContent

def loadStopWords(filepath):
  stopwords = [line.strip() for line in open(filepath,'r',encoding="utf-8").readlines()]
  return stopwords

def removeStopWords(sentence):
  stopwords = loadStopWords('D:/Topic/code/segment/stop_words.txt')
  output = ''
  sentence_seg = jieba.cut(sentence)
  for word in sentence_seg:
    if word not in stopwords:
      output += word
      output += " "
  return output    


if __name__ == '__main__':

  program = os.path.basename(sys.argv[0])
  logger = logging.getLogger(program)
  logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
  logging.root.setLevel(level=logging.INFO)
  logger.info("running %s" % ' '.join(sys.argv))

   # check and process input arguments
  #if len(sys.argv) < 3:
   # print (globals()['__doc__'] % locals())
    #sys.exit(1)
  #inp, outp = sys.argv[1:3]
  #inp = "corpus.txt"
  outp = "output_4_category.txt"
  rootdir = "D:/Topic/code/segment/data/word2vec_data/articles"

  space = " "
  i = 0
  r = '[，。：？“”！、（）《》’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+' 
  foutput = open(outp,'a',encoding='utf-8')
  count = 0
  for root, dirs, files in os.walk(rootdir):
    #finput = open (str(filenames),"r",encoding='utf-8')
    for folder in dirs:
      for root, dirs, files in os.walk(rootdir+'/'+ folder):
        for file in files:
          finput = open (os.path.join(root,file),"r",encoding='utf-8')
          for line in finput:
            line = re.sub(r,' ',line.strip())
        #line_seg = removeStopWords(jieba.cut(line))
            line_seg = removeStopWords(line)
        #foutput.write(space.join(line_seg))
            foutput.write(line_seg)
            i = i + 1
            if (i % 1000 == 0):
              logger.info("Saved " + str(i) + " articles_seg")
          foutput.write('\n')
          finput.close()
          logger.info("Finished Saved " + str(i) + " articles")
  foutput.close() 
  
