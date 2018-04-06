from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize

def RemoveStopWords(line,stopWords):
	seg_sen = ""
	for w in line:
		if w not in stopWords:
			seg_sen = seg_sen + w + " "
	return seg_sen

rootdir = "D:/Topic/code/segment/compare/dataset/stanfordSentimentTreebank/standard"
stopWords = set(stopwords.words('english'))
originFile = open(rootdir+'\\train.txt','r',encoding='utf-8').read().split('\n')
newFile = rootdir+"/train_nosw.txt"
foutput = open(newFile,'a',encoding='utf-8')
for line in originFile:
	words = word_tokenize(line)
	lineSeg = RemoveStopWords(words,stopWords)
	foutput.write(lineSeg)
	foutput.write('\n')
foutput.close()
#originFile.close()


