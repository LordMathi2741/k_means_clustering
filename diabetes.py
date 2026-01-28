from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler

dataset = load_diabetes()
data = dataset.data
columns = dataset.feature_names
df = pd.DataFrame(data=data, columns=columns)
feature_1 = 'age'
feature_2 = 'sex'
data_feature_1 = df[[feature_1]].values
data_feature_2 = df[[feature_2]].values
data_feature = np.hstack((data_feature_1, data_feature_2))

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_feature)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=None)
kmeans.fit(data_scaled)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, 
            edgecolors='black', label='Centroides')
plt.xlabel(f'{feature_1} (escalado)')
plt.ylabel(f'{feature_2} (escalado)')
plt.title(f'K-Means Clustering basado en {feature_1} y {feature_2}')
plt.legend()
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()