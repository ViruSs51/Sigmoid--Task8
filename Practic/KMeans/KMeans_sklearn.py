import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs


X, _ = make_blobs(n_samples=420, 
                  centers=3, 
                  cluster_std=0.4, 
                  random_state=0)

#plt.scatter(X[:, 0], X[:, 1], s=15)
#plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
ids = kmeans.predict(X)

#plt.scatter(X[:, 0], X[:, 1], c=ids, s=15, cmap='viridis')

centers = kmeans.cluster_centers_

#plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)
#plt.show()

from sklearn.datasets import make_moons

X, _ = make_moons(500, noise=.06, random_state=69)
kmeans = KMeans(2, random_state=69)
kmeans.fit(X)
labels = kmeans.predict(X)

#plt.scatter(X[:, 0], X[:, 1], c=labels, s=15, cmap='viridis')
#plt.show()


from sklearn.cluster import SpectralClustering

model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
assign_labels='kmeans')
labels = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=15, cmap='viridis')
plt.show()


