from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import string
import re
from gensim import utils, corpora, matutils, models
from text_processing import text_processed


def count_vectorizer_feature_vector():
    token_array = text_processed()
    vectorizer = CountVectorizer(encoding='utf-8', analyzer='word', stop_words='english', binary='false',
                                 min_df=0.01)
    # tokenize and build vocab
    vectorizer.fit_transform(token_array)
    print(vectorizer.get_feature_names())
    f_vector = vectorizer.transform(token_array)
    print(f_vector.shape)
    print(f_vector.toarray())
    return f_vector


def tf_idf_vect_feature_vector():
    token_array = text_processed()
    print(token_array)
    vectorizer = TfidfVectorizer(analyzer="word")
    vector = vectorizer.fit_transform(token_array)
    print(vectorizer.get_feature_names())
    print(vector.toarray())
    return vector


def tf_idf_trans_feature_vector():
    token_array = text_processed()
    print(token_array)
    vectorizer = TfidfTransformer(analyzer="word")
    # tokenize and build vocab
    X = vectorizer.fit_transform(token_array)
    analyze = vectorizer.build_analyzer()
    print(analyze("subject is not the case"))
    # summarize
    print(vectorizer.get_feature_names())
    # summarize encoded vector
    print(X.toarray())
    return


def word2vec_feature_vector():
    token_array = text_processed()
    print(token_array)

    model = models.Word2Vec(token_array, min_count=1)
    print(model)
    return
