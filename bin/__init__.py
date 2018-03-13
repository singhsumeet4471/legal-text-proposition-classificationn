import numpy
import scipy
import numpy as np
from xml2csv import convert_xml2csv

np.set_printoptions(threshold=np.inf)

from feature_extraction import count_vectorizer_feature_vector, tf_idf_trans_feature_vector, tf_idf_vect_feature_vector, \
    word2vec_feature_vector,cluster_indices
from clustering_function import get_cluster_kmeans,get_dbscan_cluster,tsne_reduction


vector = count_vectorizer_feature_vector()
print('tf-idf vector')
matrix = tf_idf_vect_feature_vector()


no_cluster = 4
cluster = get_cluster_kmeans(vector, no_cluster)
print(cluster)
cluster = get_cluster_kmeans(matrix, no_cluster)
print(cluster)
cluster = get_dbscan_cluster(matrix,1)
print(cluster)
print(cluster)

# word2vec_feature_vector()


