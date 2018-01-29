from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['author1', 'author2', 'author3']
__emails__ = ['fatherchristmoas@northpole.dk', 'toothfairy@blackforest.no', 'easterbunny@greenfield.de']


def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append(l.lower().split())
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class mSkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        # winSize: Size of th window
        # minCount : minimum times word appears

        self.winSize = winSize
        self.minCount = minCount

        self.vocabulary = {}
        for sentence in sentences:
            for word in sentence:
                if word not in vocabulary:
                    self.vocabulary[word] = 1
                else:
                    self.vocabulary[word] += 1
        self.length_vocabulary = len(self.vocabulary)
        self.vocabulary_list = list(vocabulary)

    def word_to_vec(self, word):

        word_vec = np.zeros(self.length_vocabulary)
        word_vec[vocabulary_list.index('word')] = 1
        return word_vec

    def context_to_matrix(self, sentence, word):

        position = sentence.index(word)

        context_matrix = word_to_vec(word)

        for context_word in sentence:
            pos_context_word = sentence.index(context_word)

            if np.abs(pos_context_word - position) <= int(self.winSize / 2) and np.abs(pos_context_word - position) > 0:
                context_matrix = np.c_[context_matrix, word_to_vec(context_word)]
        context_matrix = context_matrix[:, 1:]
        return context_matrix

        raise NotImplementedError('implement it!')

    def train(self, stepsize, epochs):
        raise NotImplementedError('implement it!')

    def save(self, path):
        raise NotImplementedError('implement it!')

    def similarity(self, word1, word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        raise NotImplementedError('implement it!')

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    vector = np.zeros(100)
    context = np.zeros((100, 10))

    f_ = sum(np.log(sigmoid(np.dot(vector, context)))) + sum(np.log(sigmoid(-np.dot(vector, context))))

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mySkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mSkipGram.load(opts.model)
        for a, b, _ in pairs:
            print sg.similarity(a, b)
