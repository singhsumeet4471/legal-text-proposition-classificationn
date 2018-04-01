import numpy as np

np.set_printoptions(threshold=np.inf)

from feature_extraction import tf_idf_vect_feature_vector,compute_distance_matrix,compute_similarity_matrix
from clustering_function import linkage_algo, get_dbscan_cluster,get_cluster_kmeans,tsne_reduction,pca_reduction


# vector = count_vectorizer_feature_vector()
# print('tf-idf vector')



# print(cluster)

# word2vec_feature_vector()

# print(dmt)

# hc = HClust(dmt)
# cluster = HClust.clusters

# web_scraping()

#

# TESTING K MEANS

# no_cluster = 4
#
test_token_array, vec, vec_matrix = tf_idf_vect_feature_vector()
# km = get_cluster_kmeans(vec_matrix, no_cluster)
test_tfidf_matrix= vec.transform(test_token_array)
# test_tfidf_matrix = truncate_SVD(test_tfidf_matrix, no_cluster)
# tested_cluster_list = km.predict(test_tfidf_matrix)
# trained_cluster_list = km.labels_.tolist()
# test_label_count = len(tested_cluster_list)
# print(trained_cluster_list)
# print(tested_cluster_list.tolist())
# clust_eval(trained_cluster_list[:test_label_count], tested_cluster_list)

cluster = get_dbscan_cluster(vec_matrix,0.5,30)
cluster = get_cluster_kmeans(vec_matrix,6)
# print(cluster)

matrix = compute_distance_matrix(vec_matrix,test_tfidf_matrix,'cosine')
linkage_algo(matrix)
matrix =compute_similarity_matrix(vec_matrix,test_tfidf_matrix)
cmatrix = compute_distance_matrix(vec_matrix,test_tfidf_matrix,'manhatten')
pca_reduction(cmatrix,6)
tsne_reduction(cmatrix)
linkage_algo(matrix)
