from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import string
import re
from text_processing import text_processed

def count_vectorizer_feature_vector():
    token = text_processed()
    str_array = re.split(r'[,.]', token)
    print(str_array)
    vectorizer = CountVectorizer()
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
    # print(vectorizer.vocabulary_)
    return