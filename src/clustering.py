import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from typing import Tuple, Optional


def perform_kmeans(
    features: np.ndarray, n_clusters: int = 10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Perform K-means clustering"""
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(features)
    silhouette_avg = silhouette_score(features, clusters)
    return clusters, kmeans.cluster_centers_, silhouette_avg


def perform_dbscan(
    features: np.ndarray, eps: float = 0.5, min_samples: int = 5
) -> Tuple[np.ndarray, int, int]:
    """Perform DBSCAN clustering"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    return clusters, n_clusters, n_noise


def detect_anomalies(
    features: np.ndarray, method: str = "kmeans", **kwargs: Optional[dict]
) -> np.ndarray:
    """Detect anomalies using clustering-based approach"""
    if method == "kmeans":
        clusters, centroids, _ = perform_kmeans(features, **kwargs)
        distances = np.linalg.norm(features - centroids[clusters], axis=1)
        anomaly_threshold = np.percentile(distances, 95)
        return distances > anomaly_threshold

    if method == "dbscan":
        clusters, _, _ = perform_dbscan(features, **kwargs)
        return clusters == -1

    raise ValueError(f"Unsupported clustering method: {method}")
