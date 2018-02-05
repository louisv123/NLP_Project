import skipGram

path = 'C:/Users/Louis/Documents/AM2014-2015-2016/2017-2018/Essec-Centrale_Paris/NLP/NLP_Project/Project_1/input-1000.txt'


sentences = skipGram.text2sentences(path)

skipmodel = skipGram.mySkipGram(sentences)
