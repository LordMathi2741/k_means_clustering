import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
import numpy as np

try:
    df = pd.read_csv('./datasets/db.csv',encoding='latin-1')
    feature = 'Release'
    data_feature = df[[feature]].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_feature)
    kmeans = KMeans(n_clusters=3, n_init=40, random_state=None)
    kmeans.fit(data_scaled)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    plt.figure(figsize=(10, 6))
    plt.scatter(data_scaled[:, 0], np.zeros_like(data_scaled[:, 0]), c
                =labels, cmap='viridis', alpha=0.6)
    plt.scatter(centers[:, 0], np.zeros_like(centers[:, 0]), c='red', marker='X', s=200, 
                    edgecolors='black', label='Centroides')
    plt.xlabel(f'{feature} (escalado)')
    plt.title(f'K-Means Clustering basado en {feature}')
    plt.legend()
    plt.colorbar(label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.show()
except FileNotFoundError:
    raise FileNotFoundError("No se encontrol el conjunto de datos 'db.csv'.")