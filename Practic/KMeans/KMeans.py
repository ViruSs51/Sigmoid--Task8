from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_blobs


X, _ = make_blobs(n_samples=420, 
                  centers=3, 
                  cluster_std=0.4, 
                  random_state=0)

def find_clusters(X: np.ndarray, 
                  n_cluesters: int, 
                  rseed: int=69
                  ) -> tuple:
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_cluesters]
    centers = X[i]

    while True:
        #E-Step
        labels = pairwise_distances_argmin(X, centers)

        #M-Step
        new_centers = np.array(
            [
                X[labels == i].mean(0) for i in range(n_cluesters)
            ]
        )

        if np.all(centers == new_centers): break

        centers = new_centers

    return centers, labels


#centers, labels = find_clusters(X, 3)

#plt.scatter(X[:, 0], X[:, 1], c=labels, s=15, cmap='viridis')
#plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)
#plt.show()

#centers, labels = find_clusters(X, 3, 55)

#plt.scatter(X[:, 0], X[:, 1], c=labels, s=15, cmap='viridis')
#plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)

#plt.show()


#silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

def draw_silhouette_plot(X, n_clusters):
    clusterer = KMeans(n_clusters=n_clusters, random_state=69)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)

    # Calculează scorurile Silhouette pentru fiecare eșantion
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # Scorul Silhouette variază de la -1 la 1
    plt.xlim([-1, 1])

    # (n_clusters + 1) * 10 este pentru inserarea unui spațiu gol între siluete
    # pentru fiecare cluster individual, pentru a le demarca clar.
    plt.ylim([0, len(X) + (n_clusters + 1) * 10])
    plt.yticks([])  # Elimină etichetele de pe axa y
    plt.xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Linia verticală pentru scorul mediu Silhouette al tuturor valorilor
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)

        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        # Calculează noul y_lower pentru următorul plot
        y_lower = y_upper + 10  # 10 pentru eșantioanele cu valoarea 0

    plt.title(f'Diagrama Silhouette pentru n_clusters = {n_clusters}')
    plt.legend(['Scor Silhouette'] + [f'Cluster {i}' for i in range(n_clusters)])
    plt.show()

draw_silhouette_plot(X, int(input('Introduceți numărul de clustere: ')))