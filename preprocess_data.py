#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:03:22 2018

@author: suvasama
"""

#------------------------------------------------------------------------------

# IMPORT PACKAGES AND FUNCTIONS

from data_utilities import read_data, preprocess_text
from data_utilities import find_frequency, find_bigrams, find_cumFrequency
import math

#------------------------------------------------------------------------------

# UNIGRAM ANALYSIS

def load_unigrams():
    # tokenize data by sentences
    tokens, vocab = read_data('Dataset2.txt')

    # clean up data
    tokens = preprocess_text(tokens)

    # split into training and test sets
    tr_tokens = tokens[math.floor(len(tokens)/2):]
    te_tokens = tokens[:math.floor(len(tokens)/2)]

    # find frequency: positive: 0; negative: 1
    trY, trX = find_frequency(tr_tokens, vocab)
    teY, teX = find_frequency(te_tokens, vocab)
    
    return trX, teX, trY, teY


#------------------------------------------------------------------------------

# BIGRAM ANALYSIS

def load_bigrams():
    tokens, vocab = read_data('Dataset2.txt')
    tokens = preprocess_text(tokens)

    tr_tokens = tokens[math.floor(len(tokens)/2):]
    te_tokens = tokens[:math.floor(len(tokens)/2)]
        
    # get the bigrams by sentence and the corresponding vocabulary
    tr_bigrms, vocab2 = find_bigrams(tr_tokens)
    te_bigrms, vocab2 = find_bigrams(te_tokens)
    
    # find frequency of bigrams and add to the data
    trY, trX = find_cumFrequency(tr_tokens, tr_bigrms, vocab, vocab2)
    teY, teX = find_cumFrequency(te_tokens, te_bigrms, vocab, vocab2)
    
    return trX, teX, trY, teY
    
