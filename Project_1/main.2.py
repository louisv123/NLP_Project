import skipGram
import numpy as np
import pandas as pd

path = '/home/quentinbb/class/nlp/NLP_Project/Project_1/input_bible.txt'
stopwords_path = '/home/quentinbb/class/nlp/NLP_Project/Project_1/stopwords.csv'
#C:/Users/Louis/Documents/AM2014-2015-2016/2017-2018/Essec-Centrale_Paris/NLP/NLP_Project/Project_1/input-1000.txt

#load the stopwords file
stopwords = skipGram.loadStopwords(stopwords_path)

#load the text and get the sentences
sentences = skipGram.text2sentences(path, stopwords)


#initialize the skipgram
skipmodel = skipGram.mySkipGram(sentences, stopwords)

#train it
skipmodel.train(1, 1)

#print some similarity to check the algo
#do only the first 50 2-uplets to check
limit = 100 #how many 2-uplet similarity you want to print
counter = 0 

for a in skipmodel.vocabulary_list:
	for b in skipmodel.vocabulary_list:
		print(a, b, skipmodel.similarity(a, b))

		counter +=1

		if counter >limit:
			break

