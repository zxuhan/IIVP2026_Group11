# KEN3238 — Mixed Latin/Devanagari Digit Classification

Kaggle challenge for the **KEN3238 Intro to Image & Video Processing** course.
Each image is a handwritten digit in **either Latin (0–9) or Devanagari (०–९)** script.
The model predicts the **digit value (0–9)** regardless of script.

A small PyTorch CNN trained from scratch reaches **≥99% validation accuracy** in ~10 epochs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the competition data from Kaggle and place it next to `train.py`:

```
project/
├── train.py
├── train/train/{0..9}/*.png     # 17,000 images, 1,700 per class
├── test/test/*.png              # 3,000 unlabeled images
├── test.csv                     # Id column — submission row order
└── sample_submission.csv        # reference header
```

## Run

```bash
python train.py
```

This will:

1. Pick a device (CUDA → MPS → CPU).
2. Load `train/train/` via `ImageFolder`, 90/10 train/val split (seeded).
3. Train the CNN for 10 epochs with AdamW + cosine LR schedule.
4. Save the best-val checkpoint to `best_model.pt`.
5. Run inference on `test/test/` in the order given by `test.csv`.
6. Write `submission.csv` (3,000 rows: `Id,Category`) ready for Kaggle upload.

A run on Apple-silicon MPS finishes in a few minutes.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | End-to-end pipeline: data → model → train → infer → submit. |
| `notebook.ipynb` | Same pipeline with per-step narrative and sample visualizations. |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Excludes data, checkpoints, and the generated submission. |
