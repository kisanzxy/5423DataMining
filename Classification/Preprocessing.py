"""
Created on Thu Oct 11 19:23:05 2018
@author: Xuyang Zhang, Xiang Li
For the purpose of the Lab#1 for CSE 5243 Autumn 2018
"""

import os
import sys
import string
import nltk

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from string import maketrans

# Class Node and trie are based on Nick Stanisha's work:
# https://nickstanisha.github.io/2015/11/21/a-good-trie-implementation-in-python.html
class Node:
    def __init__(self, label=None, data=None):
        self.label = label
        self.data = data
        self.children = dict()  # empty dictionary          
        self.wordFreq = dict()   # empty dictionary: store teh frequency of each word inside a specific article
    
    # There are multiple possiblites, key is either just a lable or a Node
    def addChild(self, key, data=None):
        if not isinstance(key, Node):   # if key is not a instance of class Node
            self.children[key] = Node(key, data)
        else:
            self.children[key.label] = key    
    
    def __getitem__(self, key):      # Get the value corresponding to the label: key
        return self.children[key]

class Trie:
    def __init__(self):
        self.head = Node()  # This is the root
        self.document = []       # a list of dictionaries
      
    
    def __getitem__(self, key):
        return self.head.children[key]
    
    def add(self, word, articleID=0):
        current_node = self.head
        word_finished = True
        
        for i in range(len(word)):
            if word[i] in current_node.children:
                current_node = current_node.children[word[i]]
            else:
                word_finished = False
                break
        
        # For ever new letter, create a new child node
        if not word_finished:
            while i < len(word):
                current_node.addChild(word[i])
                current_node = current_node.children[word[i]]
                i += 1
                current_node.data = word
                # Let's store the full word at the end node only when it does not exit before
        
        if articleID in current_node.wordFreq:
            current_node.wordFreq[articleID] += 1
        else:
            current_node.wordFreq[articleID] = 1
        
    
    def has_word(self, word):
        if word == '':
            return False
        if word == None:
            raise ValueError('Trie.has_word requires a not-Null string')
        
        # Start at the top
        current_node = self.head
        exists = True
        for letter in word:
            if letter in current_node.children:
                current_node = current_node.children[letter]
            else:
                exists = False
                break
        
        # Still need to check if we just reached a word like 'T', 
		# 'T' is just an indicator for a tage which is added by ourself
        # that isn't actually a full word in our dictionary
        if exists:
            if current_node.data == None:
                exists = False
        
        return exists
    
    def start_with_prefix(self, prefix):
        """ Returns a list of all words in tree that start with prefix """
        words = list()
        if prefix == None:
            raise ValueError('Requires not-Null prefix')
        
        # Determine end-of-prefix node
        top_node = self.head
        for letter in prefix:
            if letter in top_node.children:
                top_node = top_node.children[letter]
            else:
                # Prefix not in tree, go no further
                return words
        
        # Get words under prefix
        if top_node == self.head:
            queue = [node for key, node in top_node.children.iteritems()]
        else:
            queue = [top_node]
        
        # Perform a breadth first search under the prefix
        while queue:
            current_node = queue.pop()
            if current_node.data != None:
            
                words.append(current_node.data)
            
            queue = [node for key,node in current_node.children.iteritems()] + queue
        
        return words
    
    def getData(self, word):
        """ This returns the 'data' of the node identified by the given word """
        if not self.has_word(word):
            raise ValueError('{} not found in trie'.format(word))
        
        # Race to the bottom, get data
        current_node = self.head
        for letter in word:
            current_node = current_node[letter]
        
        return current_node.data
    
        
    # get the frequency of a word in the article with the ID: articleID
    def getFreq(self, word, articleID):
        """ This returns the 'data' of the node identified by the given word """
        if not self.has_word(word):
            raise ValueError('{} not found in trie'.format(word))
        
        # Race to the bottom, get data
        current_node = self.head
        for letter in word:
            current_node = current_node[letter]
        
        return current_node.wordFreq[articleID]
    
    # the total frequency of a word through all the documents
    def getTotalFreq(self, word):
        """ This returns the 'data' of the node identified by the given word """
        if not self.has_word(word):
            raise ValueError('{} not found in trie'.format(word))
        
        # Race to the bottom, get data
        current_node = self.head
        for letter in word:
            current_node = current_node[letter]
        
        freqSum = 0
        for val in current_node.wordFreq.values():
            freqSum += val
        
        return freqSum

###############################################################################
################ function(s) for generating document objects ##################
###############################################################################

class Article_Doc:
    def __init__(self):
        self.topics = []
        self.places = []
        self.title = []
        self.body = []
    
    # all text parts are the raw string contents from the data directly
    def set_doc(self, raw_article):
        # if an article does not have body, then won't be counted
        if raw_article.body == None:
            return False
        else:
            body = raw_article.body.text.encode('utf-8', 'ignore')
            if body:
                self.body = tokenize(body)
        
        if raw_article.topics != None:
            topics = raw_article.topics.text.encode('utf-8', 'ignore')
            if topics:
                self.topics = tokenize(topics, 1)
                
        
        if raw_article.places != None:
            places = raw_article.places.text.encode('utf-8', 'ignore')
            if places:
                self.places = tokenize(places, 1)
       
        if raw_article.title != None:
            title = raw_article.title.text.encode('utf-8', 'ignore')
            if title:
                self.title = tokenize(title)
        
        return True
        
    
    def get_term(self, term):
        if term == 'topics':
            return self.topics
        elif term == 'places':
            return self.places
        elif term == 'title':
            return self.title
        elif term == 'body':
            return self.body
    

######################## helper function   ###################################
####### help tokenize all the meaningful words inside the data   #############
####### text is the string type contents of an article #############
def tokenize(text, shortest = 2):               
    
    ascii = (text.lower()).decode('utf-8', 'ignore').encode('Utf-8', 'ignore')
    no_digits = ascii.translate(None, string.digits)   # (table, deletetable): to remove all the digits
    no_punctuation = no_digits.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)    # for all the words and parenthesis, they are separated as isolated tokens
    no_stop_words = [w for w in tokens if not w in stopwords.words('english')]
    # filter out non-english words
    eng = [y for y in no_stop_words if wordnet.synsets(y)]
    # lemmatization process
    lemmas = []
    lmtzr = WordNetLemmatizer()
    for token in eng:
        lemmas.append(lmtzr.lemmatize(token))
    # stemming process
    stems = []
    stemmer = PorterStemmer()
    for token in lemmas:
        stems.append(stemmer.stem(token).encode('utf-8','ignore'))
    
    meaningful_words = [w for w in stems if len(w) > shortest]   # the word length should be greater than 1, this includes: w != '/' and w != 'T' 

    return meaningful_words

########################  helper function  ################################

# The text passed to generate_doc should be the raw text
# also all the tabluate should be in the raw data format
def generate_doc(text, separator, trie, articleID):
    article_Doc_List = []      
    beauDoc = BeautifulSoup(text.lower(), "html.parser")
    # loop through all the articles except the last one (a blank one)
    for raw_article in beauDoc.find_all(separator.lower()):        
        single_article = Article_Doc() 
        # if the article body is not empty
        if single_article.set_doc(raw_article):
            articleID += 1
            # here we focus on both the title and the body part of an article
            token_list = single_article.title + single_article.body
        
            for word in token_list:
                trie.add(word, articleID)
 
            article_Doc_List.append(single_article)
    
    return article_Doc_List, articleID


def read_documents(trie):
    Total_Doc = []
    separator = 'REUTERS'
    
    articleID = 0
    for file in os.listdir('data'):
        data = open(os.path.join(os.getcwd(), "data", file), 'r')
        text = data.read()
        data.close()
        file_doc, articleID = generate_doc(text, separator, trie, articleID)   # each time update the articleID between different files
        print articleID
        
        Total_Doc += file_doc
    
    # print Total_Doc
    return Total_Doc



