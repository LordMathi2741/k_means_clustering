from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

x1 = np.random.normal(2,0.5,(50,3))
x2 = np.random.normal(7,0.5,(50,3))
x3 = np.random.normal(4,0.5,(50,3))
data = np.vstack((x1,x2,x3))

k=3
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(data[:,0], data[:,1], c=labels)
plt.scatter(centers[:,0], centers[:,1],s=200, marker='x')
plt.title('K-Means Clustering')
plt.show()