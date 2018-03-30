import numpy as np
import pandas as pd
from gensim import models
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances

from text_processing import text_processed

pd.set_option("display.max_columns", 100)
np.set_printoptions(threshold=np.inf)


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
    #print("token: ", token_array)
    vectorizer = TfidfVectorizer(stop_words='english', analyzer="word")
    #print(vectorizer)
    matrix = vectorizer.fit_transform(token_array)
    data_frame = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())
    #print(data_frame)
    return matrix


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
    return X


def word2vec_feature_vector():
    token_array = text_processed()
    print(token_array)
    model = models.Word2Vec(token_array, min_count=1)
    print(model)
    return


def cluster_indices(cluster_assignments):
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])
    return indices

def dissimalrity_matrix():
    token_array = text_processed()
    vectorizer = TfidfVectorizer(stop_words='english', analyzer="word")

    td_if = vectorizer.fit_transform(token_array)
    x= td_if.toarray()
    y= vectorizer.get_feature_names()
    print(x)
    print(y)
    matrix = euclidean_distances(td_if)

    #print(matrix)
    return matrix

