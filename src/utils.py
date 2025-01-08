import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize_anomalies(images, labels, anomalies, output_path, max_to_display=10):
    """Visualize a subset of images flagged as anomalies."""
    anomaly_indices = np.where(anomalies == 1)[0]
    num_anomalies = len(anomaly_indices)

    if num_anomalies == 0:
        raise Exception("No anomalies to visualize.")

    if num_anomalies > max_to_display:
        anomaly_indices = random.sample(list(anomaly_indices), max_to_display)

    cols = 5
    rows = (len(anomaly_indices) // cols) + (len(anomaly_indices) % cols > 0)
    _, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for i, idx in enumerate(anomaly_indices):
        img = images[idx]
        label = labels[idx]
        axes[i].imshow(img)
        axes[i].set_title(f"Anomaly ID: {idx}\nLabel: {label}")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def visualize_latent_space(
    features: np.ndarray, clusters: np.ndarray, save_path: Optional[str] = None
) -> None:
    """Visualize latent space using t-SNE"""
    tsne = TSNE(n_components=2)
    latent_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=clusters, cmap="viridis")
    plt.title("Latent Space Clustering")
    plt.colorbar(scatter)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def save_anomaly_report(
    features: np.ndarray, labels: np.ndarray, anomalies: np.ndarray, save_path: str
) -> None:
    """Save detailed anomaly detection report"""
    with open(save_path, "w") as f:
        f.write("Anomaly Detection Report\n")
        f.write("=====================\n\n")
        f.write(f"Total Samples: {len(features)}\n")
        f.write(f"Anomalies Detected: {np.sum(anomalies)}\n\n")

        f.write("Anomaly Details:\n")
        for i, (is_anomaly, label) in enumerate(zip(anomalies, labels)):
            if is_anomaly:
                f.write(f"Sample {i}: Label {label}\n")
