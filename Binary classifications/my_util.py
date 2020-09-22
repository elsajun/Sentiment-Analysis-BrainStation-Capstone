import pandas as pd
import numpy as np 

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string

from nltk.corpus import stopwords 
stemmer = nltk.stem.PorterStemmer()
ENGLISH_STOP_WORDS = stopwords.words('english')
ENGLISH_STOP_WORDS.remove('not')
ENGLISH_STOP_WORDS.remove('no')

def my_tokenizer(sentence):
    '''
        This function accepts sentence. Then we clean the text data.
        The steps are: punctuation, tokenize, stopwords, and stemming. 
    '''
    
    for punctuation_mark in string.punctuation:
        # Remove punctuation and set to lower case
        sentence = sentence.replace(punctuation_mark,'').lower()

    # split sentence into words
    listofwords = sentence.split(' ')
    listofstemmed_words = []
    
        
    # Remove stopwords and any tokens that are just empty strings
    for word in listofwords:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Stem words
            stemmed_word = stemmer.stem(word)
            listofstemmed_words.append(stemmed_word)

    return listofstemmed_words




def train_test_split(X,y):
    '''
        This function accepts X as the independent variables and y as the dependent variable.
        For this project, the X is the words on customer reviews, and y is the product's rating
    '''
    
    from sklearn.model_selection import train_test_split

    # Taking a chuck for our 25% test set
    X_remainder, X_test, y_remainder, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    # Splitting the remainder in two chunks
    X_train, X_validation, y_train, y_validation = train_test_split(X_remainder, 
                                                                    y_remainder, 
                                                                    stratify=y_remainder, test_size=0.25, random_state=42)
    from sklearn.feature_extraction.text import CountVectorizer 
    bagofwords = CountVectorizer(tokenizer=my_tokenizer)
    bagofwords.fit(X_train)
    X_train = bagofwords.transform(X_train)
    X_validation = bagofwords.transform(X_validation)
    X_test = bagofwords.transform(X_test)
    X_remainder = bagofwords.transform(X_remainder)
    
    #oversampling the minority class
    
    from imblearn.over_sampling import RandomOverSampler, SMOTE
#     smote = SMOTE(random_state=777,k_neighbors=5)
#     X_train, y_train = smote.fit_sample(X_train, y_train)
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    
    return X_train, X_validation, X_test, X_remainder, y_train, y_validation, y_test, y_remainder, bagofwords


def text_pipeline(X_train, y_train, new_text):
    '''
        This function accepts X_train, y_train, and new_text(list of 2 reviews)
        This function will make predictions of the sentiment analysis of the new_text 
    '''
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import LinearSVC
    clf_linearSVC_pipe = Pipeline([("vect", CountVectorizer()), 
                                ("tfidf", TfidfTransformer()),
                                ("clf_linearSVC", LinearSVC())])
    clf_linearSVC_pipe.fit(X_train, y_train)

    from sklearn.model_selection import GridSearchCV
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],    
                'tfidf__use_idf': (True, False)} 
    gs_clf_LinearSVC_pipe = GridSearchCV(clf_linearSVC_pipe, parameters, n_jobs=-1)
    gs_clf_LinearSVC_pipe = gs_clf_LinearSVC_pipe.fit(X_train, y_train)

    return(gs_clf_LinearSVC_pipe.predict(new_text))

    