from data_import import data_txt_import_array
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import string
import re
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def text_processed():
    txt = data_txt_import_array('test.txt')
    # txt = strip_punctation(txt)
    format_token = text_stop_words(txt)
    stemmed_token = token_stemmer(format_token)
    lemmeted_token = token_lemmetizer(format_token)
    str = ' '.join(lemmeted_token)
    str_array = re.split(r'[,.]', str)
    # print(txt)
    # print('filtered_sentence \n')
    # print(format_token)
    # stemming
    # print('Stemmed token \n')
    # print(stemmed_token)
    # lemm
    # print('Lemmeted token \n')
    # print(lemmeted_token)
    # print('Lemmeted string \n')
    # print(str)
    return str_array


def text_stop_words(unformat_text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(unformat_text)
    print('tokens \n')
    print(word_tokens)

    format_token = []
    for w in word_tokens:
        if w not in stop_words:
            format_token.append(w)

    return format_token


def strip_punctation(text):
    translate_table = dict((ord(char), None) for char in string.punctuation)
    stripped_text = text.translate(translate_table)
    return stripped_text


def token_stemmer(token):
    stemmer = PorterStemmer()
    stemmed_token = []
    for w in token:
        stemmed_token.append(stemmer.stem(w))
    return stemmed_token


def token_lemmetizer(token):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmeted_token = []
    for w in token:
        lemmeted_token.append(wordnet_lemmatizer.lemmatize(w, pos='v'))
    return lemmeted_token


def split_data(data, train_split=0.8):
    data = np.array(data)
    num_train = data.shape[0] * train_split
    npr.shuffle(data)

    return (data[:num_train], data[num_train:])
