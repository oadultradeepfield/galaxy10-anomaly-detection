import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from typing import Tuple


def create_feature_extractor() -> nn.Sequential:
    """Create a feature extractor based on ResNet18."""
    pretrained_model = models.resnet18(weights="DEFAULT")
    feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor


def extract_features(
    dataloader: DataLoader, model: nn.Module, device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from a given dataloader using the feature extractor."""
    model.to(device)
    features, labels = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images).squeeze()
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    return np.vstack(features), np.hstack(labels)
