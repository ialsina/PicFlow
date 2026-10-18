from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def compute_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, list[float]]:
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_.tolist()
    return coords, ev


def compute_umap(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(15, max(2, len(X) - 1)),
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def compute_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    n_clusters = min(n_clusters, len(X))
    if n_clusters < 1:
        return np.zeros(len(X), dtype=int)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(X)
