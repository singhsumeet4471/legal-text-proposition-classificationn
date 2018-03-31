import numpy as np

np.set_printoptions(threshold=np.inf)

from feature_extraction import tf_idf_vect_feature_vector,compute_distance_matrix
from clustering_function import linkage_algo

# vector = count_vectorizer_feature_vector()
# print('tf-idf vector')

# cluster = get_dbscan_cluster(matrix,1)
# print(cluster)
# print(cluster)

# word2vec_feature_vector()

# txt = data_csv_import('20180313151844.csv')
# vec = dissimalrity_matrix()
# print(vec)

# dmt = DistanceMatrix1(matrix)

# print(dmt)

# hc = HClust(dmt)
# cluster = HClust.clusters

# web_scraping()

#

# TESTING K MEANS
test_token_array, vec, vec_matrix = tf_idf_vect_feature_vector()
# no_cluster = 4
#
# km = get_cluster_kmeans(vec_matrix, no_cluster)
test_tfidf_matrix= vec.transform(test_token_array)
# test_tfidf_matrix = truncate_SVD(test_tfidf_matrix, no_cluster)
# tested_cluster_list = km.predict(test_tfidf_matrix)
# trained_cluster_list = km.labels_.tolist()
# test_label_count = len(tested_cluster_list)
# print(trained_cluster_list)
# print(tested_cluster_list.tolist())
# clust_eval(trained_cluster_list[:test_label_count], tested_cluster_list)

matrix = compute_distance_matrix(vec_matrix,test_tfidf_matrix)
linkage_algo(matrix)
