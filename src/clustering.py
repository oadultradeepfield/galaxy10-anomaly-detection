from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def perform_kmeans(
    features: np.ndarray, n_clusters: int = 10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Perform K-means clustering"""
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(features)
    return clusters, kmeans.cluster_centers_


def perform_dbscan(
    features: np.ndarray, eps: Optional[float] = None, min_samples: int = 5
) -> Tuple[np.ndarray, int, int]:
    """Perform DBSCAN clustering with simple eps estimation"""
    if eps is None:
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(features)
        distances, _ = nn.kneighbors(features)

        eps = np.median(distances[:, -1])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    return clusters, n_clusters, n_noise


def detect_anomalies(
    features: np.ndarray, method: str = "kmeans", **kwargs: Optional[dict]
) -> np.ndarray:
    """Detect anomalies using clustering-based approach"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    if method == "kmeans":
        clusters, centroids = perform_kmeans(scaled_features, **kwargs)
        distances = np.linalg.norm(features - centroids[clusters], axis=1)
        q1 = np.percentile(distances, 25, axis=0)
        q3 = np.percentile(distances, 75, axis=0)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return (distances < lower_bound) | (distances > upper_bound)

    if method == "dbscan":
        clusters, _, _ = perform_dbscan(scaled_features, **kwargs)
        return clusters == -1

    raise ValueError(f"Unsupported clustering method: {method}")
