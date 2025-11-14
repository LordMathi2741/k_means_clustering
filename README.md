# Clustering Practice with scikit-learn

## Project overview
This repository contains simple clustering experiments using scikit-learn. The goal is to practice common clustering algorithms, compare results, and visualize cluster assignments on two example data sources.

## Goals
- Explore KMeans, DBSCAN, and Agglomerative Clustering.
- Compare algorithm behavior on synthetic vs. real data.
- Provide reproducible examples and minimal plotting code.

## Requirements
- Python 3.8+
- scikit-learn
- matplotlib (optional, for plots)
Install:
```
pip install scikit-learn matplotlib
```

## Example sources

1. Synthetic blobs (easy, controlled clusters)
- Generator: sklearn.datasets.make_blobs
- Use case: verify algorithm recovers known cluster centers and test sensitivity to n_clusters and initialization.

Minimal example:
```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
km = KMeans(n_clusters=4, random_state=42).fit(X)
plt.scatter(X[:,0], X[:,1], c=km.labels_, cmap='tab10', s=10)
plt.title('KMeans on synthetic blobs')
plt.show()
```

2. Iris dataset (real, labeled for evaluation)
- Source: sklearn.datasets.load_iris
- Use case: unsupervised clustering compared to species labels (adjusted rand score, homogeneity).

Minimal example:
```python
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

iris = load_iris()
X = iris.data
cl = AgglomerativeClustering(n_clusters=3).fit(X)
print("ARI vs true labels:", adjusted_rand_score(iris.target, cl.labels_))
```

## Suggested experiments
- Compare KMeans, DBSCAN, and Agglomerative on both sources.
- Vary parameters: n_clusters, eps/min_samples (DBSCAN), linkage (Agglomerative).
- Evaluate with silhouette score and adjusted rand index where labels available.

## Project structure (suggested)
- data/            # optional saved datasets
- notebooks/       # exploratory notebooks and plots
- src/             # scripts for running experiments
- README.md

## Notes
- Standardize features (StandardScaler) before clustering when scales differ.
- Visualize results in 2D via PCA or t-SNE for higher-dimensional data.

License: MIT (or choose appropriate)  