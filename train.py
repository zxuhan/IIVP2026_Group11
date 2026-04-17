"""KEN3238 — Mixed Latin/Devanagari handwritten-digit classification.

Single-file pipeline: load data → train CNN → predict → write submission.csv.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

SEED = 42
DATA_DIR = Path(__file__).resolve().parent
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"

IMG_SIZE = 32
BATCH_SIZE = 128
VAL_FRACTION = 0.10
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
CKPT_PATH = DATA_DIR / "best_model.pt"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = DATA_DIR / "submission.csv"


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


class TestDataset(Dataset):
    """Loads test images in the order given by test.csv's Id column."""

    def __init__(self, ids: list[int], img_dir: Path, transform: transforms.Compose):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        with Image.open(self.img_dir / f"{img_id}.png") as im:
            im = im.convert("L")
            return self.transform(im), img_id


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, loss_fn,
                    device: torch.device) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main() -> None:
    set_seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    full, train_loader, val_loader = build_loaders(device)
    print(f"classes (folder → label): {full.class_to_idx}")
    print(f"total images: {len(full)}  train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")
    counts = Counter(label for _, label in full.samples)
    print(f"per-class counts: {dict(sorted(counts.items()))}")

    model = Net().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:02d}/{EPOCHS}  lr={lr_now:.2e}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"val loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ↳ saved best checkpoint to {CKPT_PATH.name} (val acc={val_acc:.4f})")

    print(f"best val accuracy: {best_val_acc:.4f}")

    # ---- Inference on test set --------------------------------------------------
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    test_ids = pd.read_csv(TEST_CSV)["Id"].tolist()
    _, eval_tf = build_transforms()
    test_ds = TestDataset(test_ids, TEST_DIR, eval_tf)
    pin = device.type == "cuda"
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=pin)

    predictions: dict[int, int] = {}
    with torch.no_grad():
        for x, ids in test_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().tolist()
            for img_id, pred in zip(ids.tolist(), preds):
                predictions[img_id] = pred

    # Verify format matches sample_submission.csv and write in test.csv order.
    with open(SAMPLE_SUB, newline="") as f:
        sample_header = next(csv.reader(f))
    assert sample_header == ["Id", "Category"], f"unexpected header: {sample_header}"

    with open(SUBMISSION_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Category"])
        for img_id in test_ids:
            writer.writerow([img_id, predictions[img_id]])

    assert len(predictions) == len(test_ids) == 3000, (
        f"prediction count mismatch: {len(predictions)} preds, {len(test_ids)} ids"
    )
    print(f"wrote {SUBMISSION_PATH} ({len(test_ids)} rows)")


if __name__ == "__main__":
    main()
