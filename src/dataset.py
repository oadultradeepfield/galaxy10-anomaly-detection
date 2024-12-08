from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from astroNN.datasets import load_galaxy10

IMG_SIZE = 128
BATCH_SIZE = 32
RANDOM_SEED = 202412

torch.manual_seed(RANDOM_SEED)


def get_transforms() -> transforms.Compose:
    """Image preprocessing transformations"""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class GalaxyDataset(Dataset):
    """PyTorch Dataset for Galaxy10 images."""

    def __init__(
        self, images: np.ndarray, labels: np.ndarray, transform: transforms.Compose
    ) -> None:
        self.images = images.astype(np.uint8)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        """Dataset size"""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a dataset sample"""
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_galaxy_dataset(test_size: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    """Load Galaxy10 and split data"""
    images, labels = load_galaxy10()
    transform = get_transforms()
    dataset = GalaxyDataset(images, labels, transform=transform)
    train_size = int(len(dataset) * (1 - test_size))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader
