from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
data = iris.data
df = pd.DataFrame(data, columns=iris.feature_names)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(data_scaled)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='x', c='red')
plt.title('K-Means Clustering on Iris Dataset')
plt.show()