"""FusionDataset: pairs each FER-2013 image with its landmark vector + mask.

The image side uses torchvision.ImageFolder with the EXACT transforms from
cnn_gap_aug_labelsmooth/data_utils.py (imported, not re-declared, so they can
never drift). The landmark side comes from the precomputed alignment cache
(CNN/fusion_gated/cache/align_<split>.npz), built once by build_alignment.py and
indexed 1:1 with ImageFolder's sample order.

Each item is (image_tensor, lm_tensor(26,), mask_tensor(scalar), label(int64)).
Images where dlib failed get a zero lm vector and mask=0.0.

Same seeded 10% val split (seed 42, random_split) as the other CNNs, so the val
set is identical across models.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

# Reuse the improved CNN's transforms verbatim.
sys.path.insert(0, os.path.join(_REPO_ROOT, "CNN", "cnn_gap_aug_labelsmooth"))
import data_utils as gap_data  # noqa: E402

CLASSES = gap_data.CLASSES
CACHE_DIR = os.path.join(_HERE, "cache")


class FusionDataset(Dataset):
    """Wraps an ImageFolder and attaches landmark vector + mask per index.

    `transform` is applied to the PIL image (use gap_data._train_tf or _eval_tf).
    The landmark/mask/label arrays are loaded from the alignment cache and must
    be the same length as the ImageFolder.
    """

    def __init__(self, split, transform):
        if split == "train":
            img_dir = os.path.join(_REPO_ROOT, "data", "train")
            cache = os.path.join(CACHE_DIR, "align_train.npz")
        elif split == "test":
            img_dir = os.path.join(_REPO_ROOT, "data", "test")
            cache = os.path.join(CACHE_DIR, "align_test.npz")
        else:
            raise ValueError(split)

        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"{cache} not found — run `python -m CNN.fusion_gated.build_alignment` first.")

        # No transform on the ImageFolder itself; we transform in __getitem__ so a
        # train/eval Subset can share one underlying folder if desired.
        self.folder = datasets.ImageFolder(img_dir)
        assert self.folder.classes == CLASSES, self.folder.classes
        self.transform = transform

        z = np.load(cache)
        self.lm = torch.from_numpy(z["lm"]).float()        # (N, 26)
        self.mask = torch.from_numpy(z["mask"]).float()    # (N,)
        cache_label = z["label"].astype(np.int64)
        assert len(self.lm) == len(self.folder), \
            f"cache N={len(self.lm)} != ImageFolder N={len(self.folder)}"
        # Cache labels must match ImageFolder targets exactly (alignment proof).
        folder_targets = np.array(self.folder.targets, dtype=np.int64)
        assert np.array_equal(cache_label, folder_targets), \
            "alignment cache label order disagrees with ImageFolder — rebuild cache."

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, i):
        img, label = self.folder[i]            # PIL image, int label
        img = self.transform(img)
        lm = self.lm[i]
        mask = self.mask[i].view(1)            # (1,) so it broadcasts in fusion
        return img, lm, mask, torch.tensor(label, dtype=torch.long)


def get_loaders(batch_size=64, val_split=0.1, num_workers=2, seed=42):
    """train_loader, val_loader, classes — same split logic as the GAP model.

    Train uses augmentation transforms, val uses eval transforms, and the split
    is seeded identically (seed 42, random_split) so the val set matches.
    """
    train_full = FusionDataset("train", gap_data._train_tf)
    eval_full = FusionDataset("train", gap_data._eval_tf)

    n_val = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(len(train_full)), [n_train, n_val], generator=g)

    train_ds = Subset(train_full, list(train_idx))
    val_ds = Subset(eval_full, list(val_idx))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, CLASSES


def get_test_loader(batch_size=64, num_workers=2):
    test_ds = FusionDataset("test", gap_data._eval_tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False), CLASSES
