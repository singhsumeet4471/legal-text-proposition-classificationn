import numpy as np
from matplotlib import pyplot as plt
from plotting_clusters import plot_histo, plot
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


def get_cluster_kmeans(tfidf_matrix, num_clusters):
    tfs_embedded = truncate_SVD(tfidf_matrix, num_clusters)
    km = KMeans(n_clusters=num_clusters, init='k-means++')
    km.fit(tfs_embedded)

    x = metrics.silhouette_score(tfidf_matrix, km.labels_, metric='euclidean')
    print("Silhouette Coefficient score for K-means is : ", x)
    plot(tfs_embedded,km,'K-Means','true')
    plot_histo(km.labels_,num_clusters,'DB-Scan')


    return km


def get_dbscan_cluster(tfidf_matrix, epsilon,samples):
    #for values less than zero scatter plot is shown which in not acceptable
    #and for values 1 and more than it only one cluster is shown which again fails to classify different label which we need
    db = DBSCAN(eps=epsilon, min_samples=samples).fit(tfidf_matrix)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    x = metrics.silhouette_score(tfidf_matrix, labels, metric='euclidean')

    print("Silhouette Coefficient score for DB-Scan is : ", x)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    tfs_embedded = truncate_SVD(tfidf_matrix, n_clusters_)
    plot(tfs_embedded, db,'DB-Scan','false')
    plot_histo(labels, n_clusters_,'DB-Scan')
    return labels


def multidim_scaling(similarity_matrix, n_components):
    one_min_sim = 1 - similarity_matrix
    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=4)
    pos = mds.fit_transform(one_min_sim)  # shape (n_components, n_samples)
    x_pos, y_pos = pos[:, 0], pos[:, 1]
    return (x_pos, y_pos)


def pca_reduction(similarity_matrix, n_components):
    one_min_sim = 1 - similarity_matrix
    pca = PCA(n_components=10)
    pos = pca.fit_transform(one_min_sim)
    x_pos, y_pos = pos[:, 0], pos[:, 1]
    return (x_pos, y_pos)


def tsne_reduction(similarity_matrix):
    one_min_sim = 1 - similarity_matrix
    tsne = TSNE(learning_rate=1000).fit_transform(one_min_sim)
    x_pos, y_pos = tsne[:, 0], tsne[:, 1]
    return (x_pos, y_pos)

def truncate_SVD(tfidf_matrix,num_clusters):
    tfs_reduced = TruncatedSVD(n_components=num_clusters, random_state=0).fit_transform(tfidf_matrix)
    tfs_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(tfs_reduced)
    return  tfs_embedded


def linkage_algo(X):
    Z = linkage(X, 'ward')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)
    plt.title('Ward-Dendrogram')
    plt.show()

    Z = linkage(X, 'single')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)
    plt.title('Single-Dendrogram')
    plt.show()

    Z = linkage(X, 'complete')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)
    plt.title('Complete-Dendrogram')
    plt.show()
