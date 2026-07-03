"""Shared data loading for FER-2013 (ImageFolder layout)."""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# FER-2013 single-channel normalization (dataset-wide mean/std, common values).
_MEAN, _STD = 0.5077, 0.2550

_train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),          # faces are roughly symmetric
    transforms.RandomRotation(10),              # mild pose jitter
    transforms.ToTensor(),
    transforms.Normalize([_MEAN], [_STD]),
])

_eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([_MEAN], [_STD]),
])


def get_loaders(data_dir, batch_size=64, val_split=0.1, num_workers=2, seed=42):
    """Return train_loader, val_loader, class names.

    A val_split fraction is carved out of data/train for checkpoint selection.
    The validation subset uses eval transforms (no augmentation).
    """
    train_dir = os.path.join(data_dir, "train")

    full_train = datasets.ImageFolder(train_dir, transform=_train_tf)
    full_eval = datasets.ImageFolder(train_dir, transform=_eval_tf)
    assert full_train.classes == CLASSES, full_train.classes

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(len(full_train)), [n_train, n_val], generator=g)

    train_ds = torch.utils.data.Subset(full_train, list(train_idx))
    val_ds = torch.utils.data.Subset(full_eval, list(val_idx))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, CLASSES


def get_test_loader(data_dir, batch_size=64, num_workers=2):
    test_dir = os.path.join(data_dir, "test")
    test_ds = datasets.ImageFolder(test_dir, transform=_eval_tf)
    assert test_ds.classes == CLASSES, test_ds.classes
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False), CLASSES
