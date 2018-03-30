import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np


def plot(data,algo):
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    label_colours = colors[algo.labels_]
    plt.scatter(data[:, 0], data[:, 1], c=label_colours)
    #centers = np.array(algo.cluster_centers_)
    #center_colors = colors[:len(centers)]
    #print(centers)
    #plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, c='black')
    plt.plot()
    plt.show()

def plot_histo(label,no_cluster):
    plt.hist(label, bins=no_cluster)
    plt.show()


