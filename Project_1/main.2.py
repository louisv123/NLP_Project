import skipGram
import numpy as np

path = 'C:/Users/Louis/Documents/AM2014-2015-2016/2017-2018/Essec-Centrale_Paris/NLP/NLP_Project/Project_1/input-1000.txt'


sentences = skipGram.text2sentences(path)

skipmodel = skipGram.mySkipGram(sentences)

skipmodel.train(1, 1)

print(skipmodel.vocabulary_list)

"""

np.dot(skipmodel.W_2[, :], skipmodel.W_1).index(max(np.dot(skipmodel.W_2[index_word1, :], skipmodel.W_1)))
"""
