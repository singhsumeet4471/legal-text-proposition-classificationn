import numpy
import scipy
import numpy as np

np.set_printoptions(threshold=np.inf)

from feature_extraction import count_vectorizer_feature_vector, tf_idf_vect_feature_vector, \
    word2vec_feature_vector

from data_import import data_csv_import
from skbio import DistanceMatrix

from sklearn.metrics.pairwise import euclidean_distances


#vector = count_vectorizer_feature_vector()
# print('tf-idf vector')
#matrix = tf_idf_vect_feature_vector()



no_cluster = 9
#cluster = get_cluster_kmeans(matrix, no_cluster)
#print(cluster)
# cluster = get_cluster_kmeans(matrix, no_cluster)
# print(cluster)
# cluster = get_dbscan_cluster(matrix,1)
# print(cluster)
# print(cluster)

# word2vec_feature_vector()

txt = data_csv_import('20180313151844.csv')
vec = tf_idf_vect_feature_vector()
#print(vec)
matrix =euclidean_distances(vec)
print(matrix)
#dmt = DistanceMatrix1(matrix)

print(dmt)

#hc = HClust(dmt)
#cluster = HClust.clusters


