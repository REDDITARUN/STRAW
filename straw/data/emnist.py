"""EMNIST data loading utilities -- transforms, datasets, and dataloaders."""

from __future__ import annotations

from typing import Dict, Optional

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

def get_transforms() -> Dict[str, transforms.Compose]:
    """Return a dictionary of named transform pipelines.

    Keys: ``"standard"``, ``"rotated"``, ``"blurred"``.
    """
    standard = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    rotated = transforms.Compose([
        transforms.RandomRotation(60),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    blurred = transforms.Compose([
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return {"standard": standard, "rotated": rotated, "blurred": blurred}


# ──────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────

def get_datasets(
    data_root: str = "./data",
    train_subset: float = 1.0,
) -> Dict[str, object]:
    """Load EMNIST-balanced train and test datasets.

    Parameters
    ----------
    data_root : str
        Directory where EMNIST data is stored / downloaded.
    train_subset : float
        Fraction of the training set to use (e.g. 0.1 for 10%).
        Defaults to 1.0 (full dataset).

    Returns
    -------
    dict with keys:
        ``"train"`` -- training dataset (possibly subsetted)
        ``"test_standard"`` -- clean test set
        ``"test_rotated"`` -- rotation-augmented test set
        ``"test_blurred"`` -- Gaussian-blur test set
    """
    tfms = get_transforms()

    # Training data
    train_full = torchvision.datasets.EMNIST(
        root=data_root, split="balanced", download=True, transform=tfms["standard"],
    )
    if train_subset < 1.0:
        step = max(1, int(round(1.0 / train_subset)))
        train_ds = Subset(train_full, range(0, len(train_full), step))
    else:
        train_ds = train_full

    # Test data (three variants)
    test_standard = torchvision.datasets.EMNIST(
        root=data_root, split="balanced", train=False, download=True,
        transform=tfms["standard"],
    )
    test_rotated = torchvision.datasets.EMNIST(
        root=data_root, split="balanced", train=False, download=True,
        transform=tfms["rotated"],
    )
    test_blurred = torchvision.datasets.EMNIST(
        root=data_root, split="balanced", train=False, download=True,
        transform=tfms["blurred"],
    )

    print(f"Train size: {len(train_ds):,}  |  Test size: {len(test_standard):,}")

    return {
        "train": train_ds,
        "test_standard": test_standard,
        "test_rotated": test_rotated,
        "test_blurred": test_blurred,
    }


# ──────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────

def get_dataloaders(
    datasets: Dict[str, object],
    batch_size: int = 128,
) -> Dict[str, DataLoader]:
    """Wrap dataset dict into DataLoaders.

    The ``"train"`` loader shuffles; test loaders do not.
    """
    loaders = {}
    for name, ds in datasets.items():
        shuffle = name == "train"
        loaders[name] = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle)
    return loaders
