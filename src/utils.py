import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from typing import Optional


def visualize_reconstruction(
    original: np.ndarray, reconstructed: np.ndarray, save_path: Optional[str] = None
) -> None:
    """Visualize original and reconstructed images"""
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original")
    axes[1].imshow(reconstructed.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Reconstructed")

    if save_path:
        plt.savefig(save_path)
    else:
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
