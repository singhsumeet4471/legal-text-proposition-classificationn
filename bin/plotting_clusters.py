import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.metrics import confusion_matrix

def plot(data,algo,name,is_center):
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    label_colours = colors[algo.labels_]
    plt.scatter(data[:, 0], data[:, 1], c=label_colours)
    plt.title(name)
    if(is_center == 'true'):
        centers = np.array(algo.cluster_centers_)
        center_colors = colors[:len(centers)]
        print(centers)
        plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, c='black')


    plt.plot()
    plt.show()

def plot_histo(label,no_cluster,name):
    plt.hist(label, bins=no_cluster)
    plt.title(name + '-histogram')
    plt.show()


# plot_confusion_matrix(trained_cluster[:test_label_count], tested_cluster_list.tolist())

def plot_confusion_matrix(training_labels, predicted_labels):
    cm = confusion_matrix(training_labels, predicted_labels)

    # Plot confusion matrix
    plt.imshow(cm, interpolation='none', cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, z, ha='center', va='center')
    plt.xlabel("kmeans label")
    plt.ylabel("truth label")
    plt.show()



