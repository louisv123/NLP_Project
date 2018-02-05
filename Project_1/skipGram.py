from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['Louis_Veillon, Quentin_Boutoille-Blois']
__emails__ = ['b00727589@essec.edu', 'b00527749@essec.edu']


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


class mySkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        # winSize: Size of th window
        # minCount : minimum times word appears
        # number of words display in context in answer
        """ code added:"""

        self.winSize = winSize
        self.minCount = minCount
        self.learning_rate = 0.17
        self.sentences = sentences
        self.nEmbed = nEmbed
        self.set_vocab()
        # weights of the firts hidden layer
        self.W_1 = np.random.rand(self.length_vocabulary, self.nEmbed)

        # weights of the second hidden layers
        self.W_2 = np.random.rand(self.length_vocabulary, self.nEmbed)

    def set_vocab(self):
        self.vocabulary = {}
        for sentence in self.sentences:
            for word in sentence:
                if word not in self.vocabulary:
                    self.vocabulary[word] = 1
                else:
                    self.vocabulary[word] += 1
        self.length_vocabulary = len(self.vocabulary)
        self.vocabulary_list = list(self.vocabulary)

    def word_to_vec(self, word):

        word_vec = np.zeros(self.length_vocabulary)
        word_vec[vocabulary_list.index('word')] = 1
        return word_vec

    def generate_D(self):

        self.Dictionary_D = {}

        for sentence in self.sentences:
            for word in sentence:

                position = sentence.index(word)

                if word not in self.Dictionary_D:
                    self.Dictionary_D[word] = []
                for context_word in sentence:
                    pos_context_word = sentence.index(context_word)

                    if np.abs(pos_context_word - position) <= int(self.winSize / 2) and np.abs(pos_context_word - position) > 0:
                        self.Dictionary_D[word].append(context_word)

    def generate_D_prime(self):

        self.Dictionary_D_prime = {}

        for sentence in self.sentences:
            for word in sentence:
                word_context_list = np.random.choice(self.vocabulary_list, 4)

                if word not in self.Dictionary_D_prime:
                    self.Dictionary_D_prime[word] = []
                for word_context in word_context_list:
                    self.Dictionary_D_prime[word].append[word_context]

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def train(self, stepsize, epochs):

        self.generate_D()
        self.generate_D_prime()

        for word in self.vocabulary_list:
            for word_context in self.Dictionary_D[word]:
                word_set = [(word, 1)] + [(word_neg, 0) for word_neg in self.vocabulary_list.sample(4)]

                for wor, label in word_set:

                    self.W_2[:, context_word] += self.learning_rate * (label - sigmoid(np.dot(self.W_1[wor, :], self.W_2[:, word_context])) * W_1[wor, :])

                    self.W_1[wor, :] += self.learning_rate * (label - sigmoid(np.dot(self.W_1[wor, :], self.W_2[:, word_context])) * W_2[:, word_context])

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

        sg = mySkipGram.load(opts.model)
        for a, b, _ in pairs:
            print(sg.similarity(a, b))


# test
# test my branch
