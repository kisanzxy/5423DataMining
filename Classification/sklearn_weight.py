# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:10:16 2018

@author: XiangLi
"""

import math
import sys
import os

from sklearn.feature_extraction.text import TfidfVectorizer

class Weights_sklearn:
    def __init__(self, Total_Doc):
        self.words = []
        self.weights = None
        self.weightFactors = dict()
        self.produce_weights(Total_Doc)

    def produce_weights(self, Total_Doc): 
        contents_dict = dict()
        # each article the main contents are joined by spaces
        for i, article in enumerate(Total_Doc):
            contents_dict[i] = ' '.join(article.title + article.body)

        tfidf_handler = TfidfVectorizer()
        weights_matrix = tfidf_handler.fit_transform(contents_dict.values())
        raw_words = tfidf_handler.get_feature_names()
        
        self.words = [word.encode('utf-8', 'ignore') for word in raw_words]
        self.weights = weights_matrix.toarray()
    
        # the first index of the weightFactors is zero or not
        for articleID, row in enumerate(self.weights):
            self.weightFactors[articleID] = dict()
            ### zeros are automatically filled here
            for i, word in enumerate(self.words):
                self.weightFactors[articleID][word] = self.weights[articleID][i]