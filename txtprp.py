# Module to do various preprocessing

# Vectorizer Options
# 1. Count Vectorizer
# 2. TF-IDF Vectorizer
# 3. Inverse Function (Adi to find)
# 4. Word2Vec

# Stemming / Text Preprocess
# 1. Stemming
# 2. Lemmatization
# 3. Stop words removal
# 4. Slicing (first n chars)
# 5. Special chars removal
# 6. Number replacement

import numpy as np
import pandas as pd
import nltk, re
from nltk import word_tokenize
from sklearn import CountVectorizer, TfidfVectorizer

class TextObject(pd.DataFrame):
    """
    A collection of strings for processing. Is a DataFrame with specific columns. 
    """

    def __init__(self):
        super(TextObject, self).__init__(columns=
            [
                "original_text",
                "text",
                "label",
                "count_vector",
                "tfidf_vector",
                "inverse_vector",
                "inverse_vector",
                "word2vec"
            ]
        )


    def add_text(self, text_collection, labels=None):
        """
        Adds a collection of text to to the TextObject. Accepts an iterable or array-like of all text strings. Optionally, accepts labels of the same length.
        Modifies self to be a pandas DataFrame with column "text" and potentially "labels".
        """
        self["original_text"] = text_collection
        self["text"] = self.original_text
        if labels:
            if len(labels)==len(self):
                self["label"] = labels
    
    def count_vectorize(self, **kwargs):
        """
        Adds a column "count_vector" showing the results of running a CountVectorizer.fit_transform on the self.text column.
        Also stores the CountVectorizer as self.count_vect. Accepts kwargs for CountVectorizer.
        """
        if kwargs:
            self.count_vect = CountVectorizer(**kwargs)
        else:
            self.count_vect = CountVectorizer()
        self["count_vector"] = self.count_vect.fit_transform(self.text)

    def tfidf_vectorize(self, **kwargs):
        """
        Same as "count_vectorize," but for Tf-Idf vectorzation.
        """
        if kwargs:
            self.tfidf = TfidfVectorizer(**kwargs)
        else:
            self.tfidf = TfidfVectorizer()
        self["tfidf_vector"] = self.tfidf.fit_transform(self.text)
    
    def invrs_vectorize(self, **kwargs):
        """
        TBU for inverse function Adi mentioned
        """
        self.inverse = None
        self["inverse_vector"] = None
    
    def wrd2v_vectorize(self, **kwargs):
        """
        TBU for Word2Vect
        """
        self.word2vec = None
        self["word2vec"] = None
    
    def vectorize_all(self):
        """
        Calls all vectorizers without args.
        """
        self.count_vectorize()
        self.tfidf_vectorize()
        self.invrs_vectorize()
        self.wrd2v_vectorize()
    
    def text_processing(self, func):
        """
        Applies a generic text processing function to the self.text field.
        """
        self.text = func(self.text)

    #Next steps:
    #   1. Use `text_processing` to add more text preprocessing functionality.
    #   2. Implement Word2Vec & inverse
    #   3. Consider adding a "history" property which shows all text transformations applied
    #   4. Consider adding a "reset" function to reset self.text to self.original_text.
    #   5. Error checking etc.

