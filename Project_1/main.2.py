import skipGram
import numpy as np
import pandas as pd

path = '/home/quentinbb/class/nlp/NLP_Project/Project_1/input_bible.txt'
#C:/Users/Louis/Documents/AM2014-2015-2016/2017-2018/Essec-Centrale_Paris/NLP/NLP_Project/Project_1/input-1000.txt

sentences = skipGram.text2sentences(path)

skipmodel = skipGram.mySkipGram(sentences)

skipmodel.train(1, 1)

print(skipmodel.vocabulary_list)


#pairs = skipGram.loadPairs(path)

#data = pd.read_csv(path, delimiter='\t')

#print(data)
# for word1, word2 in data
# pairs = zip(data[word1], data[word2])

for a in skipmodel.vocabulary_list:
	for b in skipmodel.vocabulary_list:
		print(a, b, skipmodel.similarity(a, b))

#print(skipmodel.similarity("eternal", "teachers"))



"""

np.dot(skipmodel.W_2[, :], skipmodel.W_1).index(max(np.dot(skipmodel.W_2[index_word1, :], skipmodel.W_1)))
"""
