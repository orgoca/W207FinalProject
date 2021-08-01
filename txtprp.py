from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import nltk as nltk
import re
import gensim 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextObject():
    """
    A collection of strings for processing. Is a DataFrame with specific 
    columns. 
    """

    def __init__(self):
        self.df = pd.DataFrame(columns=[
            "original_text",
            "text",
            "label"
        ]
        )
        self.applied_transformations = []
        self.count_vect = None
        self.count_vect_array = None
        self.tfidf_vect = None
        self.tfidf_vect_array = None
        self.wv_vect = None
        self.wv_vect_array = None
        nltk.download("stopwords")
        nltk.download("punkt")

    def add_text(self, text_collection, labels=None):
        """
        Adds a collection of text to to the TextObject. Accepts an iterable or 
        array-like of all text strings. Optionally, accepts labels of the same 
        length. Modifies self to be a pandas DataFrame with column "text" and 
        potentially "labels".
        """
        self.df["original_text"] = text_collection
        self.df["text"] = self.df.original_text
        self.applied_transformations = []
        if type(labels) == type(None):
            return
        if len(labels) == len(self.df):
            self.df["label"] = labels

    def count_vectorize(self, **kwargs):
        """
        Adds a column "count_vector" showing the results of running a 
        CountVectorizer.fit_transform on the self.text column. Also stores the 
        CountVectorizer as self.count_vect. Accepts kwargs for CountVectorizer.
        """
        if kwargs:
            self.count_vect = CountVectorizer(**kwargs)
        else:
            self.count_vect = CountVectorizer()
        self.count_vect_array = self.count_vect.fit_transform(self.df.text)

    def tfidf_vectorize(self, **kwargs):
        """
        Same as "count_vectorize," but for Tf-Idf vectorzation.
        """
        if kwargs:
            self.tfidf_vect = TfidfVectorizer(**kwargs)
        else:
            self.tfidf_vect = TfidfVectorizer()
        self.tfidf_vect_array = self.tfidf_vect.fit_transform(self.df.text)

    def wrd2v_vectorize(self, **kwargs):
        """
        Creates a word2vec model with kwargs parameters. Also creates sentence
        vectors using the mean vector scores of the words in the sentence. 
        """
        self.wv_vect = gensim.models.word2vec.Word2Vec(sentences = self.df.text, **kwargs)
        features = []
        for tokens in self.df.text:
            zero_vector = np.zeros(self.wv_vect.vector_size)
            vectors = []
            for token in tokens:
                if token in self.wv_vect.wv:
                    try:
                        vectors.append(self.wv_vect.wv[token])
                    except KeyError:
                        continue
            if vectors:
                vectors = np.asarray(vectors)
                avg_vec = vectors.mean(axis=0)
                features.append(avg_vec)
            else:
                features.append(zero_vector)
        self.wv_vect_array = features

    def vectorize_all(self):
        """
        Calls all vectorizers without args.
        """
        self.count_vectorize()
        self.tfidf_vectorize()
        self.wrd2v_vectorize()

    # Text Processing Methods:

    def text_processing(self, function):
        """
        Applies a generic text processing function to the self.text field.
        """
        self.df.text = self.df.text.apply(function)
        self.applied_transformations.append(function.__name__)

    def lower(self):
        self.df.text = self.df.text.str.lower()
        self.applied_transformations.append("lower")
    
    def strip(self):
        self.df.text = self.df.text.str.strip()
        self.applied_transformations.append("strip")

    def remove_single_digits(self):
        self.df.text = self.df.text.apply(lambda text: re.sub("([\d]+)", "", text))
        self.applied_transformations.append("remove_single_digits")
    
    def remove_nonletter_chars(self):
        self.df.text = self.df.text.apply(lambda text: re.sub("[^A-Za-z0-9 \\n]", " ", text))
        self.applied_transformations.append("remove_nonletter_chars")

    def stop_word_tokenize(self):
        def tokenize(text):
            stoplist = nltk.corpus.stopwords.words('english')
            finalTokens = []
            tokens = nltk.word_tokenize(text)
            for w in tokens:
                if (w not in stoplist):
                    finalTokens.append(w)
            text = " ".join(finalTokens)
            return text
        self.df.text = self.df.text.apply(tokenize)
        self.applied_transformations.append("stop_word_tokenize")
    
    def stem_sentence(self):
        def stem(text):
            porter=PorterStemmer()
            token_words=nltk.tokenize.word_tokenize(text)
            token_words
            stem_sentence=[]
            for word in token_words:
                stem_sentence.append(porter.stem(word))
                stem_sentence.append(" ")
            return "".join(stem_sentence)
        self.df.text = self.df.text.apply(stem)
        self.applied_transformations.append("stem_sentence")

    def lemmatize_sentence(self):
        def lem(text):
            wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
            #token_words
            token_words=nltk.tokenize.word_tokenize(text)
            lemm_sentence=[]
            for word in token_words:
                lemm_sentence.append(wordnet_lemmatizer.lemmatize(word))
                lemm_sentence.append(" ")
            return "".join(lemm_sentence)
        self.df.text = self.df.text.apply(lem)
        self.applied_transformations.append("lemmatize_sentence")

    def process_all(self, lower=True, strip=True, remove_single_digits =True, 
        remove_nonletter_chars=True, stop_word_tokenize=True, stem_sentence=True,
        lemmatize_sentence=True):
        if lower:
            self.lower()
        if strip:
            self.strip()
        if remove_single_digits:
            self.remove_single_digits()
        if remove_nonletter_chars:
            self.remove_nonletter_chars()
        if stop_word_tokenize:
            self.stop_word_tokenize()
        if stem_sentence:
            self.stem_sentence()
        if lemmatize_sentence:
            self.lemmatize_sentence()
