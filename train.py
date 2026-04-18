"""KEN3238 — Mixed Latin/Devanagari handwritten-digit classification.

Single-file pipeline: load data → train CNN (×N seeds with EMA) → ensemble+TTA → write submission.csv.
"""

from __future__ import annotations

import copy
import math
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
SEEDS = [42]
DATA_DIR = Path(__file__).resolve().parent
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"

IMG_SIZE = 32
BATCH_SIZE = 128
VAL_FRACTION = 0.10
EPOCHS = 60
WARMUP_EPOCHS = 5
LR = 3e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
EMA_DECAY = 0.999
CKPT_FMT = "best_model_seed{seed}.pt"
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
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15),
                                scale=(0.85, 1.15), shear=10),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def build_tta_transforms() -> list[transforms.Compose]:
    normalize = transforms.Normalize((0.5,), (0.5,))
    base = [transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE))]
    specs = [
        None,
        {"degrees": (-5, -5)},
        {"degrees": (5, 5)},
        {"degrees": (-10, -10)},
        {"degrees": (10, 10)},
        {"degrees": 0, "translate": (0.08, 0.0)},
        {"degrees": 0, "scale": (0.95, 0.95)},
        {"degrees": 0, "scale": (1.05, 1.05)},
    ]
    out: list[transforms.Compose] = []
    for s in specs:
        ops = list(base)
        if s is not None:
            ops.append(transforms.RandomAffine(**s))
        ops += [transforms.ToTensor(), normalize]
        out.append(transforms.Compose(ops))
    return out


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
    """Wraps a Subset and applies a transform at __getitem__ time."""

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


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class Net(nn.Module):
    """BN + residual CNN with GAP head. ~300k params."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResBlock(32, 64), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(ResBlock(64, 128), nn.MaxPool2d(2))
        self.stage3 = ResBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


class EMA:
    """Exponential moving average of model parameters + buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.step = 0
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        # Warm up EMA so early steps aren't dominated by random init (timm-style).
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            if b.dtype.is_floating_point:
                s.mul_(d).add_(b.detach(), alpha=1 - d)
            else:
                s.copy_(b)


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
                              num_workers=2, pin_memory=pin, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=pin, persistent_workers=True)
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


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    return x_mixed, y, y[idx], lam


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, scheduler,
                    loss_fn, device: torch.device, ema: EMA) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_mix, ya, yb, lam = mixup_batch(x, y, MIXUP_ALPHA)
        optimizer.zero_grad()
        logits = model(x_mix)
        loss = lam * loss_fn(logits, ya) + (1 - lam) * loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (lam * (pred == ya).float() + (1 - lam) * (pred == yb).float()).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def build_scheduler(optimizer, steps_per_epoch: int):
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = EPOCHS * steps_per_epoch
    cosine_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / cosine_steps
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_seed(seed: int, device: torch.device) -> float:
    print(f"\n===== seed {seed} =====")
    set_seed(seed)
    full, train_loader, val_loader = build_loaders(device)
    if seed == SEEDS[0]:
        print(f"classes (folder → label): {full.class_to_idx}")
        print(f"total images: {len(full)}  train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")
        counts = Counter(label for _, label in full.samples)
        print(f"per-class counts: {dict(sorted(counts.items()))}")

    model = Net().to(device)
    if seed == SEEDS[0]:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model params: {n_params:,}")

    ema = EMA(model, decay=EMA_DECAY)
    ema.shadow.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, steps_per_epoch=len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    ckpt_path = DATA_DIR / CKPT_FMT.format(seed=seed)
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler,
                                          loss_fn, device, ema)
        val_loss, val_acc = evaluate(ema.shadow, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:02d}/{EPOCHS}  lr={lr_now:.2e}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"ema val loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ema.shadow.state_dict(), ckpt_path)
            print(f"  ↳ saved best EMA checkpoint to {ckpt_path.name} (val acc={val_acc:.4f})")
    print(f"seed {seed} best val accuracy: {best_val_acc:.4f}")
    return best_val_acc


@torch.no_grad()
def ensemble_predict(device: torch.device) -> dict[int, int]:
    test_ids = pd.read_csv(TEST_CSV)["Id"].tolist()
    tta_tfs = build_tta_transforms()
    pin = device.type == "cuda"

    sum_probs = torch.zeros(len(test_ids), 10)
    models: list[Net] = []
    for seed in SEEDS:
        ckpt = DATA_DIR / CKPT_FMT.format(seed=seed)
        m = Net().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models.append(m)

    id_order: list[int] = []
    for tta_idx, tf in enumerate(tta_tfs):
        test_ds = TestDataset(test_ids, TEST_DIR, tf)
        loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=pin)
        offset = 0
        for x, ids in loader:
            x = x.to(device)
            batch_probs = torch.zeros(x.size(0), 10, device=device)
            for m in models:
                batch_probs += F.softmax(m(x), dim=1)
            batch_probs = batch_probs.cpu()
            sum_probs[offset:offset + x.size(0)] += batch_probs
            if tta_idx == 0:
                id_order.extend(ids.tolist())
            offset += x.size(0)
        print(f"  TTA {tta_idx + 1}/{len(tta_tfs)} done")

    preds = sum_probs.argmax(1).tolist()
    return {img_id: int(p) for img_id, p in zip(id_order, preds)}


def main() -> None:
    device = pick_device()
    print(f"device: {device}")

    val_accs: list[float] = []
    for seed in SEEDS:
        val_accs.append(train_one_seed(seed, device))
    print(f"\nper-seed best val accs: {['%.4f' % a for a in val_accs]}")
    print(f"mean best val acc: {sum(val_accs) / len(val_accs):.4f}")

    print("\n===== ensemble + TTA inference =====")
    predictions = ensemble_predict(device)

    with open(SAMPLE_SUB, newline="") as f:
        sample_header = next(csv.reader(f))
    assert sample_header == ["Id", "Category"], f"unexpected header: {sample_header}"

    test_ids = pd.read_csv(TEST_CSV)["Id"].tolist()
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
