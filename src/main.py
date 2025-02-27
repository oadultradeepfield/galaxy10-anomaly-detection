import os

import colorama
import numpy as np
from colorama import Fore, Style

from autoencoder import (compute_reconstruction_errors,
                         detect_anomalies_from_threshold_iqr,
                         set_anomaly_threshold_iqr, train_autoencoder)
from clustering import detect_anomalies
from dataset import load_galaxy_dataset
from feature_extraction import create_feature_extractor, extract_features
from utils import (save_anomaly_report, visualize_anomalies,
                   visualize_latent_space)

colorama.init(autoreset=True)


def print_header(message: str) -> None:
    """Print a stylized header with emoji."""
    print(f"\n{Fore.CYAN}🚀 {message} 🚀{Style.RESET_ALL}")


def print_success(message: str) -> None:
    """Print a success message with a green checkmark."""
    print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")


def print_info(message: str) -> None:
    """Print an informational message with a light bulb."""
    print(f"{Fore.BLUE}💡 {message}{Style.RESET_ALL}")


def main() -> None:
    print_header("Galaxy Anomaly Detection Pipeline Initiated!")
    output_dir = "anomaly_detection_results"

    os.makedirs(output_dir, exist_ok=True)
    print_info(f"Output directory created: {output_dir}")

    print_header("Loading Galaxy Dataset")
    train_loader, test_loader = load_galaxy_dataset()
    print_success(f"Loaded {len(train_loader.dataset)} training images")
    print_success(f"Loaded {len(test_loader.dataset)} test images")

    print_header("Extracting Features")
    feature_extractor = create_feature_extractor()
    train_features, _ = extract_features(train_loader, feature_extractor)
    test_features, test_labels = extract_features(test_loader, feature_extractor)
    print_success(
        f"Extracted features: Training set {train_features.shape}, Test set {test_features.shape}"
    )

    print_header("Training Autoencoder")
    autoencoder = train_autoencoder(train_features)
    print_success("Autoencoder training completed successfully!")

    print_header("Detecting Anomalies")
    reconstruction_losses = compute_reconstruction_errors(autoencoder, test_features)
    anomaly_threshold = set_anomaly_threshold_iqr(reconstruction_losses)
    reconstruction_losses_anomalies = detect_anomalies_from_threshold_iqr(
        reconstruction_losses, anomaly_threshold
    )
    kmeans_anomalies = detect_anomalies(test_features, method="kmeans")
    dbscan_anomalies = detect_anomalies(test_features, method="dbscan")
    common_anomalies = (
        reconstruction_losses_anomalies & kmeans_anomalies & dbscan_anomalies
    )

    print_success(
        f"Reconstruction Loss Anomalies Detected: {np.sum(reconstruction_losses_anomalies)}"
    )
    print_success(f"K-means Anomalies Detected: {np.sum(kmeans_anomalies)}")
    print_success(f"DBSCAN Anomalies Detected: {np.sum(dbscan_anomalies)}")
    print_success(
        f"Total Anomalies Detected (Intersection): {np.sum(common_anomalies)}"
    )

    print_header("Generating Visualizations")
    latent_space_path = os.path.join(output_dir, "latent_space.png")
    visualize_latent_space(test_features, test_labels, latent_space_path)
    print_success(f"Latent space visualization saved to {latent_space_path}")

    print_header("Saving Anomaly Reports")
    reconstruction_losses_report_path = os.path.join(
        output_dir, "reconstruction_losses_anomalies.txt"
    )
    kmeans_report_path = os.path.join(output_dir, "kmeans_anomalies.txt")
    dbscan_report_path = os.path.join(output_dir, "dbscan_anomalies.txt")

    save_anomaly_report(
        test_features,
        test_labels,
        reconstruction_losses_anomalies,
        reconstruction_losses_report_path,
    )
    save_anomaly_report(
        test_features, test_labels, kmeans_anomalies, kmeans_report_path
    )
    save_anomaly_report(
        test_features, test_labels, dbscan_anomalies, dbscan_report_path
    )

    print_info(
        f"Reconstruction Loss anomaly report saved to {reconstruction_losses_report_path}"
    )
    print_success(f"K-means anomaly report saved to {kmeans_report_path}")
    print_success(f"DBSCAN anomaly report saved to {dbscan_report_path}")

    print_header("Visualizing a Few Anomalies")
    anomaly_visualization_path = os.path.join(output_dir, "sampled_anomalies.png")
    visualize_anomalies(
        images=test_loader.dataset.dataset.images,
        labels=test_labels,
        anomalies=common_anomalies,
        output_path=anomaly_visualization_path,
        max_to_display=10,
    )

    print_header("Galaxy Anomaly Detection Pipeline Completed Successfully! 🌌")


if __name__ == "__main__":
    main()
