from data_import import data_txt_import_array
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import string
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def text_processed():
    txt = data_txt_import_array('test.txt')
    print(txt)
    # txt = strip_punctation(txt)

    format_token = text_stop_words(txt)
    print('filtered_sentence \n')
    print(format_token)
    # stemming
    stemmed_token = token_stemmer(format_token)
    print('Stemmed token \n')
    print(stemmed_token)
    # lemm
    lemmeted_token = token_lemmetizer(format_token)
    print('Lemmeted token \n')
    print(lemmeted_token)
    str = ' '.join(lemmeted_token)
    print('Lemmeted string \n')
    print(str)
    str_array = re.split(r'[,.]', str)
    print(str_array)
    vectorizer = TfidfVectorizer()
    # tokenize and build vocab
    vectorizer.fit(str_array)
    # summarize
    print(vectorizer.get_feature_names())
    # encode document
    vector = vectorizer.transform("subject is not the case")
    # summarize encoded vector
    print(vector.shape)
    print(type(vector))
    print(vector.toarray())
    #print(vectorizer.vocabulary_)
    return


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
