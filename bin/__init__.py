import numpy as np

np.set_printoptions(threshold=np.inf)

from feature_extraction import tf_idf_vect_feature_vector
from clustering_function import get_dbscan_cluster

#vector = count_vectorizer_feature_vector()
# print('tf-idf vector')
matrix = tf_idf_vect_feature_vector()


no_cluster = 5

cluster = get_dbscan_cluster(matrix,1.5,3)

#plot(cluster)

# cluster = get_cluster_kmeans(matrix, no_cluster)
# print(cluster)
# cluster = get_dbscan_cluster(matrix,1)
# print(cluster)
# print(cluster)

# word2vec_feature_vector()

#txt = data_csv_import('20180313151844.csv')
#vec = dissimalrity_matrix()
#print(vec)

#dmt = DistanceMatrix1(matrix)

#print(dmt)

#hc = HClust(dmt)
#cluster = HClust.clusters


