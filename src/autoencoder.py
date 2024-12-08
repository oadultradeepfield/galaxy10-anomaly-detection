import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 2048, latent_dim: int = 64) -> None:
        """Initialize Autoencoder model"""
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def train_autoencoder(
    train_features: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-4,
) -> nn.Module:
    """Train the autoencoder"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    autoencoder = Autoencoder(input_dim=train_features[0].shape[0])
    autoencoder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    for epoch in range(epochs):
        autoencoder.train()
        epoch_loss = 0
        for batch in train_features:
            batch = torch.tensor(batch, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_features)}")

    return autoencoder


def compute_reconstruction_errors(
    autoencoder: nn.Module,
    test_features: np.ndarray,
) -> torch.Tensor:
    """Compute reconstruction errors for test features"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder.eval()
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed_features = autoencoder(test_features_tensor)

    return (
        torch.mean((test_features_tensor - reconstructed_features) ** 2, dim=1)
        .cpu()
        .numpy()
    )


def set_anomaly_threshold(
    reconstruction_errors: np.ndarray, percentile: float = 95
) -> float:
    """Set the threshold for anomaly detection based on the reconstruction error distribution"""
    threshold = np.percentile(reconstruction_errors, percentile)
    print(f"Anomaly detection threshold (percentile {percentile}): {threshold}")
    return threshold


def detect_anomalies_from_threshold(
    reconstruction_errors: np.ndarray, threshold: float
) -> np.ndarray:
    """Detect anomalies based on reconstruction errors exceeding the threshold"""
    anomalies = reconstruction_errors > threshold
    return anomalies
