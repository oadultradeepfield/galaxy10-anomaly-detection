from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models


def load_images(paths: List[str]) -> List[Image.Image]:
    """Load the images from a list of paths and return a list of Pillow Image objects."""
    return [Image.open(path) for path in paths]
    
def resize_image(img: Image.Image) -> Image.Image:
    """Resizes the given Pillow image to 128x128 pixels."""
    return img.resize((128, 128), Image.ANTIALIAS)

def create_feature_extractor() -> nn.Sequential:
    """Create a feature extractor based on ResNet50."""
    pretrained_model = models.resnet50(weights="DEFAULT")
    feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor

def load_model(model: nn.Module, path: str = "models/autoencoder.pth") -> nn.Module:
    """Load the model from the specified path."""
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def compute_reconstruction_errors(autoencoder: nn.Module, test_features: np.ndarray) -> torch.Tensor:
    """Compute reconstruction errors for test features"""
    device = torch.device("cpu")
    autoencoder.eval()
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed_features = autoencoder(test_features_tensor)

    return torch.mean((test_features_tensor - reconstructed_features) ** 2, dim=1).cpu().numpy()
    