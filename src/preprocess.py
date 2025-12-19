import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

logger = logging.getLogger(__name__)


class CustomImageDataset(Dataset):
    """Wrapper for image datasets with preprocessing."""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if isinstance(self.dataset, (datasets.CIFAR10, datasets.CIFAR100)):
            img, label = self.dataset[idx]
        elif isinstance(self.dataset, datasets.ImageFolder):
            img, label = self.dataset[idx]
        else:
            img, label = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
        
        # CRITICAL: Return (img, label) separately - DO NOT concatenate labels to inputs
        return img, label


def get_cifar10_transforms(preprocessing_config: Optional[Dict[str, Any]] = None):
    """Create CIFAR-10 transforms with normalization and augmentation."""
    
    if preprocessing_config is None:
        preprocessing_config = {
            "normalize": True,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
            "augmentation": True,
        }
    
    mean = preprocessing_config.get("mean", [0.4914, 0.4822, 0.4465])
    std = preprocessing_config.get("std", [0.2470, 0.2435, 0.2616])
    use_augmentation = preprocessing_config.get("augmentation", True)
    
    # Defensive checks for normalization parameters
    assert len(mean) == 3 and len(std) == 3, "Mean and std must have 3 values for RGB"
    assert all(isinstance(x, (int, float)) for x in mean), "Mean values must be numeric"
    assert all(isinstance(x, (int, float)) for x in std), "Std values must be numeric"
    assert all(x >= 0 and x <= 1 for x in std), "Std values must be in [0, 1]"
    
    if use_augmentation:
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        train_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


def get_cifar100_transforms(preprocessing_config: Optional[Dict[str, Any]] = None):
    """Create CIFAR-100 transforms with normalization and augmentation."""
    
    if preprocessing_config is None:
        preprocessing_config = {
            "normalize": True,
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2675, 0.2565, 0.2761],
            "augmentation": True,
        }
    
    mean = preprocessing_config.get("mean", [0.5071, 0.4867, 0.4408])
    std = preprocessing_config.get("std", [0.2675, 0.2565, 0.2761])
    use_augmentation = preprocessing_config.get("augmentation", True)
    
    # Defensive checks for normalization parameters
    assert len(mean) == 3 and len(std) == 3, "Mean and std must have 3 values for RGB"
    assert all(isinstance(x, (int, float)) for x in mean), "Mean values must be numeric"
    assert all(isinstance(x, (int, float)) for x in std), "Std values must be numeric"
    assert all(x >= 0 and x <= 1 for x in std), "Std values must be in [0, 1]"
    
    if use_augmentation:
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        train_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


def get_dataset(
    dataset_name: str,
    cache_dir: str = ".cache/",
    preprocessing_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dataset, Dataset]:
    """Load dataset and return train and test splits."""
    
    if preprocessing_config is None:
        preprocessing_config = {"normalize": True, "augmentation": True}
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading {dataset_name} from {cache_dir}")
    
    if dataset_name == "CIFAR-10":
        train_transform, test_transform = get_cifar10_transforms(preprocessing_config)
        
        train_dataset = datasets.CIFAR10(
            root=str(cache_dir),
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = datasets.CIFAR10(
            root=str(cache_dir),
            train=False,
            download=True,
            transform=test_transform,
        )
    
    elif dataset_name == "CIFAR-100":
        train_transform, test_transform = get_cifar100_transforms(preprocessing_config)
        
        train_dataset = datasets.CIFAR100(
            root=str(cache_dir),
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = datasets.CIFAR100(
            root=str(cache_dir),
            train=False,
            download=True,
            transform=test_transform,
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    return train_dataset, test_dataset
