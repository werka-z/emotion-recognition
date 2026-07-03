"""Data loading for the ResNet-SE FER-2013 CNN (ImageFolder layout).

Comparability is preserved exactly as in cnn_gap_aug_labelsmooth:
  - same dataset (data/train, data/test),
  - same seeded 10% val split (seed 42) -> the same val/train partition as the
    baseline and cnn_gap_aug_labelsmooth, so val numbers line up,
  - same single-channel normalization constants,
  - val/test use NO augmentation.

What changes vs cnn_gap_aug_labelsmooth:
  - Slightly stronger geometric/photometric train aug (the deeper ResNet can
    absorb more aug before underfitting): wider affine, a touch of brightness/
    contrast jitter, and RandomErasing at a higher probability.
  - A MixUp/CutMix collate function (mixup_cutmix_collate) used by the TRAIN
    loader only. This is the biggest regularizer added in this model: it blends
    pairs of images/labels (MixUp) or pastes a patch of one image onto another
    (CutMix), which directly fights overfitting and FER's noisy human labels.
    The collate returns (x, y_a, y_b, lam); the loss is the lam-weighted mix of
    the two targets. lam=1 / y_b==y_a means "no mixing" for that batch.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# FER-2013 single-channel normalization (same constants as baseline / mod).
_MEAN, _STD = 0.5077, 0.2550

_train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=15, translate=(0.12, 0.12), scale=(0.85, 1.15), shear=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),    # lighting robustness
    transforms.ToTensor(),
    transforms.Normalize([_MEAN], [_STD]),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.18)),    # occlusion robustness
])

_eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([_MEAN], [_STD]),
])


def _rand_bbox(h, w, lam):
    """Random box covering (1-lam) of the area, for CutMix."""
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
    x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
    return y1, y2, x1, x2


class MixupCutmixCollate:
    """Collate_fn that applies MixUp or CutMix to each batch.

    With probability `prob` a batch is mixed; otherwise it passes through
    unmixed (lam=1, y_b=y_a). When mixing, `cutmix_share` of the time it's
    CutMix (Beta(cutmix_alpha)), else MixUp (Beta(alpha)).

    Returns batches as (x, y_a, y_b, lam) so train.py can compute the
    lam-weighted two-target loss uniformly.

    A plain class (not a closure) so it can be pickled when DataLoader sends it
    to worker processes (num_workers > 0 uses multiprocessing + pickle).
    """

    def __init__(self, alpha=0.2, cutmix_alpha=1.0, prob=0.5, cutmix_share=0.5):
        self.alpha = alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.cutmix_share = cutmix_share

    def __call__(self, batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.tensor([b[1] for b in batch])

        if np.random.rand() > self.prob:
            return xs, ys, ys, 1.0  # no mixing this batch

        index = torch.randperm(xs.size(0))
        if np.random.rand() < self.cutmix_share:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            h, w = xs.shape[2], xs.shape[3]
            y1, y2, x1, x2 = _rand_bbox(h, w, lam)
            xs[:, :, y1:y2, x1:x2] = xs[index, :, y1:y2, x1:x2]
            # adjust lam to the true pasted-area fraction
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))
        else:
            lam = float(np.random.beta(self.alpha, self.alpha))
            xs = lam * xs + (1.0 - lam) * xs[index]
        return xs, ys, ys[index], lam


def get_loaders(data_dir, batch_size=64, val_split=0.1, num_workers=2, seed=42,
                use_mixup=True):
    """Return train_loader, val_loader, class names.

    Identical split logic to baseline / cnn_gap_aug_labelsmooth (same seed -> same
    val set). The train loader uses the MixUp/CutMix collate when use_mixup is set;
    the val loader never mixes and uses eval transforms.
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

    collate = MixupCutmixCollate() if use_mixup else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, CLASSES


def get_test_loader(data_dir, batch_size=64, num_workers=2):
    test_dir = os.path.join(data_dir, "test")
    test_ds = datasets.ImageFolder(test_dir, transform=_eval_tf)
    assert test_ds.classes == CLASSES, test_ds.classes
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False), CLASSES
