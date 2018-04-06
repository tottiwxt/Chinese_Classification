
def classify(index):
	index = float(index)
	if index >= 0  and index <=0.2:
		return 0
	elif index>0.2 and index <=0.4:
		return 1
	elif index>0.4 and index <=0.6:
		return 2
	elif index>0.6 and index <=0.8:
		return 3
	elif index>0.8 and index <=1:
		return 4


data_root = 'D:\Topic\code\segment\compare\dataset\stanfordSentimentTreebank'

dictionary = open(data_root+'\dictionary.txt','r',encoding='utf-8').read().split('\n')
articles = open(data_root+'\datasetSentences.txt','r',encoding='utf-8').read().split('\n')
grade_file = open(data_root+'\sentiment_labels.txt','r',encoding='utf-8').read().split('\n')
sentenct_output = open(data_root+'\standard\sentences.txt','a',encoding='utf-8')
label_output = open(data_root+'\standard\labels.txt','a',encoding='utf-8')

sentence_phrase_output = open(data_root+'/standard/big_train.txt','a',encoding='utf-8')
sentence_phrase_label =  open(data_root+'/standard/big_label.txt','a',encoding='utf-8')


grades = []
grades_index = []
for line in grade_file:
	index , grade = line.split('|')
	grades.append(grade)
	grades_index.append(index)
'''
#print(grades_index)
sentences = []
for article in articles:
	label, sentence = article.split('	')
	sentences.append(sentence)
#print(sentences[0])

d_sentence = []
d_index = []
for line in dictionary:
	sentence , label = line.split('|')
	d_sentence.append(sentence)
	d_index.append(label)


tmp = zip(d_sentence,d_index)
for sentence in sentences:
	if sentence in d_sentence:
		index = d_index[d_sentence.index(sentence)]
		#print(index)
		if index in grades_index:
			#print(grades[grades_index.index(index)])
			sentenct_output.write(str(sentence))
			sentenct_output.write('\n')
			label_output.write(str(classify(grades[grades_index.index(index)])))
			label_output.write('\n')
'''
for line in dictionary:
	sentence , label = line.split('|')
	if label in grades_index:
		sentence_phrase_output.write(sentence)
		sentence_phrase_output.write('\n')
		sentence_phrase_label.write(str(classify(grades[grades_index.index(label)]))) 
		sentence_phrase_label.write('\n') 
sentence_phrase_label.close()
sentence_phrase_output.close()
	



