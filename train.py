"""KEN3238 — Mixed Latin/Devanagari handwritten-digit classification.

Single-file pipeline: load data → train CNN → predict → write submission.csv.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
DATA_DIR = Path(__file__).resolve().parent
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"

IMG_SIZE = 32
BATCH_SIZE = 128
VAL_FRACTION = 0.10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize((0.5,), (0.5,))
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies a transform at __getitem__ time.

    Lets us share one ImageFolder index but keep train vs. val transforms separate.
    """

    def __init__(self, subset: torch.utils.data.Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, label = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        with Image.open(img) as im:
            im = im.convert("L")
            return self.transform(im), label


class Net(nn.Module):
    """Small CNN for 32x32 grayscale digits — ~200k params, 10 logits."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        # after two 2x2 pools on 32x32 input -> 8x8 feature map
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


def build_loaders(device: torch.device):
    train_tf, eval_tf = build_transforms()
    full = datasets.ImageFolder(str(TRAIN_DIR))
    n_val = int(len(full) * VAL_FRACTION)
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_sub, val_sub = random_split(full, [n_train, n_val], generator=gen)
    train_ds = TransformSubset(train_sub, train_tf)
    val_ds = TransformSubset(val_sub, eval_tf)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=pin)
    return full, train_loader, val_loader


def main() -> None:
    set_seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    full, train_loader, val_loader = build_loaders(device)
    print(f"classes (folder → label): {full.class_to_idx}")
    print(f"total images: {len(full)}  train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")
    counts = Counter(label for _, label in full.samples)
    print(f"per-class counts: {dict(sorted(counts.items()))}")

    x, y = next(iter(train_loader))
    print(f"batch tensor: {tuple(x.shape)}  dtype={x.dtype}  label sample: {y[:8].tolist()}")

    model = Net().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        logits = model(x.to(device))
    print(f"model params: {n_params:,}  forward output: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
