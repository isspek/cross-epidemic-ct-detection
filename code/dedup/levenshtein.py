import numpy as np

import pyximport
pyximport.install()
from code.dedup import cylevenshtein

class SamenessLevenshtein():
    '''
    Decision function if two texts are the same with minor differences,
    based on Levenshtein distance.
    '''

    def __init__(self, treshold = 0):
        '''
        :param treshold: all texts with distance less or equal
            to treshold are considered the same
        '''
        self.treshold = treshold
        self.lev = LevenshteinDistance()

    def __call__(self, text1, text2):
        '''
        :return : true if texts are considered the same"
        '''
        dist = self.lev(text1, text2)
        if dist <= self.treshold: return True
        else: return False

class LevenshteinDistance():
    '''
    Lehvenstein distance between two strings, alphabet is words not letters.
    '''

    def __init__(self, txt2words = lambda txt: txt.split(), cython = False, resetDict = True):
        '''
        :param txt2words: callable that converts text to a list of words
        :param cython: if True, use faster cython function to compute distance
        :param resetDict: if True, clear wordToken->index dict for every call,
            False can lead to optimization but can result in memory increase.
            Setting is relevant only if cython == True.
        '''
        self.txt2words = txt2words
        self.cython = cython
        self.dict = {} # word -> int index
        self.resetDict = resetDict

    def __tokFilter(self, text):
        '''
        if text is a list of tokens, just return, otherwise return tokenization
        '''
        if isinstance(text, list): return text
        else: return self.txt2words(text)

    def __resetDict(self): self.dict = {}

    def __toInts(self, tokens):
        '''
        add tokens to string->int dictionary and return list of integers
            corresponding to tokens
        '''
        result = np.empty(len(tokens), dtype=np.int32); i = 0
        for t in tokens:
            if t in self.dict: result[i]=self.dict[t]
            else: # map token to index equal to number of tokens in the dict
                key = len(self.dict)
                self.dict[t] = key
                result[i]=key
            i+=1
        return result

    def __cyDistance(self, text1, text2):
        '''
        Compute Levenshtein distance between two texts.
        :param text1, text2: string or list of string tokens
        '''
        if self.resetDict: self.__resetDict()
        words1, words2 = self.__tokFilter(text1), self.__tokFilter(text2)
        ints1 = self.__toInts(words1)
        ints2 = self.__toInts(words2)
        return cylevenshtein.dist(ints1, ints2)

    def __cyDistanceMax(self, text1, text2, max):
        '''
        Compute Levenshtein distance between two texts.
        :param text1, text2: string or list of string tokens
        '''
        if self.resetDict: self.__resetDict()
        words1, words2 = self.__tokFilter(text1), self.__tokFilter(text2)
        ints1 = self.__toInts(words1)
        ints2 = self.__toInts(words2)
        return cylevenshtein.distStop(ints1, ints2, max)

    def __pyDistance(self, text1, text2):
        '''
        Compute Levenshtein distance between two texts, using only python code.
        :param text1, text2: string or list of string tokens
        '''
        words1, words2 = self.__tokFilter(text1), self.__tokFilter(text2)
        N, M = len(words1), len(words2)
        dist = np.empty((N+1, M+1), dtype=np.uint32)
        # init matrices
        for i in range(N+1): dist[i, 0] = i
        for i in range(M+1): dist[0, i] = i
        # calc levenshtein
        for i in range(1, N+1):
            for j in range(1, M+1):
                if words1[i-1] == words2[j-1]: dist[i,j] = dist[i-1,j-1]
                else:
                    dl = dist[i-1,j]+1
                    ins = dist[i,j-1]+1
                    subst = dist[i-1,j-1]+1
                    dist[i,j] = min(dl, ins, subst)
        return dist[N, M]

    def __call__(self, text1, text2, max = None):
        '''
        Return levenshtein distance beteeen two texts if max is None.
        Othervise return distance if it is smaller than max, else max+1
        :param text1: string or list of string tokens
        :param text2: string or list of string tokens
        :param max: integer or None
        :return:
        '''
        if max is None:
            if self.cython: return self.__cyDistance(text1, text2)
            else: return self.__pyDistance(text1, text2)
        else:
            return self.__cyDistanceMax(text1, text2, max)

def testLevenshtein2(s1, s2):
    #print cylevenshtein.testType(['a', 'b'])
    dist = LevenshteinDistance()
    print(dist(s1, s2))

if __name__ == '__main__':
    testLevenshtein2('a b c', 'a b c')