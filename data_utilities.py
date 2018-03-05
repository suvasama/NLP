#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:33:23 2018

@author: suvasama
"""

#------------------------------------------------------------------------------

# IMPORT PACKAGES

# natural language toolkit
import nltk, string 
from nltk.corpus import stopwords
from nltk.stem import snowball

import numpy as np


#------------------------------------------------------------------------------

# IMPORT DATA AND SPLIT INTO SENTENCES

def read_data(data):
    text = open(data).read()
    
    # clean up data
    tokenizer = nltk.TweetTokenizer()
    # separate sentences
    tokens = [tokenizer.tokenize(t) for t in nltk.sent_tokenize(text)]
    
    # remove the text preceding the class name of each sentence
    for i in range(len(tokens)):
        if 'pos' in tokens[i]:            
            while tokens[i][0] != 'pos':
                tokens[i].remove(tokens[i][0])
            
        elif 'neg' in tokens[i]:
            while tokens[i][0] != 'neg':
                tokens[i].remove(tokens[i][0])
    
    # get words and vocabulary
    vocab = nltk.word_tokenize(text)
    vocab = list(set(w.lower() for w in vocab))
    
    # remove the class names from vocabulary
    if 'pos' in vocab:
        vocab.remove('pos'); 
    elif 'neg' in vocab:
        vocab.remove('neg')

    return tokens, vocab

        
#------------------------------------------------------------------------------

# TEXT PREPROCESSING

def preprocess_text(tokens):

    wnl = nltk.WordNetLemmatizer()
    st = snowball.EnglishStemmer()

    # remove stopwords, punctuation, non-alphabetic characters; stem and lemmatize
    sentences = []
    for i in range(len(tokens)):
        words = [w for w in tokens[i] if w.lower() 
            not in stopwords.words('english') and not w in string.punctuation]
        words = [w for w in words if w.isalpha()]      
        words = [st.stem(w) for w in words]
        # need to exclude pos from lemmatization
        for j in range(len(words)):
            if words[j] != 'pos':
                words[j] = wnl.lemmatize(words[j])
        sentences.append(words)

    return sentences


#------------------------------------------------------------------------------

# FREQUENCY DISTRIBUTION
    
# compute the frequency of words for each class 
def find_frequency(tokens, vocab):
    Y = np.zeros(len(tokens)); X = np.zeros((len(tokens), len(vocab)));
    j = 0;
    for i in range(len(tokens)):
        if 'neg' in tokens[i]:
            Y[i] = 1
        for w in tokens[i]:
            if w in vocab:
                X[j,vocab.index(w)] += 1          
        j += 1
            
    return Y, X
    

#------------------------------------------------------------------------------

# BIGRAMS

def find_bigrams(tokens):
    bigrms = []
    for s in tokens:
        if 'pos' in s:
            s.remove('pos')
        elif 'neg' in s:
            s.remove('neg')
        # pairs of words from text
        pairs = list(zip(s,s[1:]))      
        bigrms.append(pairs)
        
    vocab2 = []
    for i in range(len(bigrms)):
        for pairs in bigrms[i]:
            vocab2.append(pairs)
            
    vocab2 = list(set(vocab2))
  
    return bigrms, vocab2


#------------------------------------------------------------------------------

# CUMULATIVE DENSITY INCLUDING BIGRAMS
    
def find_cumFrequency(tokens, bigrms, vocab, vocab2):
    Y = np.zeros(len(tokens)); 
    X = np.zeros((len(tokens), len(vocab) + len(vocab2)));
    j = 0
    for i in range(len(tokens)):
        if 'neg' in tokens[i]:
            Y[i] = 1
        for w in tokens[i]:
            if w in vocab:
                X[j,vocab.index(w)] += 1
                
        for pair in bigrms[i]:
            if pair in vocab2:
                X[j,len(vocab) + vocab2.index(pair)] += 1
                
        j += 1
            
    return Y, X


