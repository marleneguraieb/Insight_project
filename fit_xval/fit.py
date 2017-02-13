#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 13:50:28 2017

@author: marleneguraieb
"""
from __future__ import print_function
import glob
import os
import pandas as pd
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import types
import warnings
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import string
from spacy.en import English
import spacy
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  confusion_matrix
import mord as m
from wordcloud import WordCloud
import random

from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


parser = English()
nlp = spacy.load('en')

# Build a list of stopwords, remove 'the', because I'm interested in specificity. 

STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
#STOPLIST.remove('the')
# Remove symbols that are not alpha-numeric, replace them with a space
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]


train, test, labels_train, labels_test = train_test_split(data_text.X,
                                                          data_text.Y, 
                                                          test_size=0.20, 
                                                          random_state=42)



vectorizer = TfidfVectorizer(tokenizer=tokenizeText, ngram_range=(1,2))
clf_bow = LinearSVC()

# the pipeline to clean, tokenize, vectorize, and classify
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf_bow)])
pipe.fit(train, labels_train)
preds_bow = pipe.predict(test)

#Grid search for LinearSVC
 
parameters = {
    'clf__C': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
              1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9),
    #'vectorizer__max_features': (None, 5000, 10000, 50000),
    'vectorizer__ngram_range': ((1,1),(1, 2)),  # unigrams or bigrams
    #'vectorizer__tokenizer': (tokenizeText),
    'vectorizer__norm': ('l1', 'l2'),
 #   'clf__alpha': (0.00001, 0.000001),
 #   'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

#This takes a long time, change to true if search is good. Best fit 

search_params = False

if __name__ == "__main__" and search_params == True:
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipe.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(train, labels_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

BoW_predict = pipe.predict(data_text.X)

X_syntax = preprocessing.normalize(syntaxFeatures(data_text))


X_stacked = pd.concat([pd.DataFrame(BoW_predict,columns=['BoW']),
                       pd.DataFrame(X_syntax),
#                       pd.DataFrame(X_pl),
                       data_feat.drop('Y',axis=1).reset_index()], axis=1, join='inner')

dt_opt = DecisionTreeClassifier(criterion= 'gini',
                                min_samples_leaf = 1,
                                max_leaf_nodes = 5,
                                max_depth = None,
                                min_samples_split = 2)


train, test, labels_train, labels_test = train_test_split(X_stacked,
                                                          data_text.Y, 
                                                          test_size=0.20, 
                                                          random_state=42)
    
    
dt_opt = dt_opt.fit(train,labels_train)    
predict = dt_opt.predict(test)
print("accuracy:", accuracy_score(labels_test, predict))
    
pd.concat([X_stacked.drop('index',axis=1),data_text['Y']],axis=1).to_csv('fin_data.csv')




