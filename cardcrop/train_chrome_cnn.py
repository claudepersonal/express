#!/usr/bin/env python3
"""
Stage 1 Training: Residual CNN for Chrome Card Artifact Removal

Trains ChromeArtifactCNN to remove artificial scratches / scanner glare
while preserving foil sheen and print detail.

Supports:
- Synthetic scratch generation (no real paired data required)
- Synthetic corruptions of real clean card images
- Residual learning (predict noise, subtract from input)

Usage:
  python train_chrome_cnn.py --epochs 30 --batch-size 8 --synthetic
"""

import argparse
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Model (same residual CNN as before)
# ---------------------------------------------------------------------------

from chrome_model import ChromeArtifactCNN


# ---------------------------------------------------------------------------
# Synthetic scratch / glare generator
# ---------------------------------------------------------------------------

def add_synthetic_scratches(clean: np.ndarray) -> np.ndarray:
    """Overlay thin linear scratches and mild glare typical of Chrome scans."""
    h, w = clean.shape[:2]
    noisy = clean.copy().astype(np.float32)

    # Thin bright scratches (horizontal + vertical + slight diagonals)
    num_lines = random.randint(3, 12)
    for _ in range(num_lines):
        thickness = random.randint(1, 2)
        color = random.randint(180, 255)
        if random.random() < 0.6:
            # mostly horizontal / vertical
            if random.random() < 0.5:
                y = random.randint(0, h - 1)
                cv2.line(noisy, (0, y), (w, y), (color, color, color), thickness)
            else:
                x = random.randint(0, w - 1)
                cv2.line(noisy, (x, 0), (x, h), (color, color, color), thickness)
        else:
            # diagonal
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            cv2.line(noisy, (x1, y1), (x2, y2), (color, color, color), thickness)

    # Mild specular glare patches
    if random.random() < 0.4:
        cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
        axes = (random.randint(20, 60), random.randint(10, 30))
        overlay = noisy.copy()
        cv2.ellipse(overlay, (cx, cy), axes, random.randint(0, 180), 0, 360, (220, 220, 220), -1)
        noisy = cv2.addWeighted(noisy, 0.75, overlay, 0.25, 0)

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChromeScratchDataset(Dataset):
    def __init__(self, clean_dir: Path, patch_size: int = 256, synthetic: bool = True):
        self.clean_paths = list(clean_dir.glob("*.jpg")) + list(clean_dir.glob("*.png"))
        self.patch_size = patch_size
        self.synthetic = synthetic
        if not self.clean_paths:
            raise ValueError('clean_dir must contain real .jpg or .png card images')

    def __len__(self):
        return max(len(self.clean_paths) * 8, 64)  # oversample for training

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.clean_paths[idx % len(self.clean_paths)]
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f'cannot read training image: {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Random crop
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            img = cv2.resize(img, (max(w, self.patch_size), max(h, self.patch_size)))
            h, w = img.shape[:2]
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)
        clean_patch = img[y:y + self.patch_size, x:x + self.patch_size]

        if self.synthetic:
            noisy_patch = add_synthetic_scratches(clean_patch)
        else:
            noisy_patch = clean_patch  # placeholder for real pairs

        clean_t = torch.from_numpy(clean_patch.transpose(2, 0, 1).copy()).float() / 255
        noisy_t = torch.from_numpy(noisy_patch.transpose(2, 0, 1).copy()).float() / 255
        return noisy_t, clean_t


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device = "GPU" if device.type == "cuda" else "CPU"
    print(f"Training on {log_device}")

    clean_dir = Path(args.clean_dir)
    if not clean_dir.is_dir():
        raise ValueError('clean_dir must be an existing folder of clean card images')

    dataset = ChromeScratchDataset(clean_dir, patch_size=args.patch_size, synthetic=args.synthetic)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ChromeArtifactCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch:03d}/{args.epochs}  L1 loss: {avg:.4f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt = out_dir / f"chrome_cnn_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  saved {ckpt}")

    final = out_dir / "chrome_cnn_final.pth"
    torch.save(model.state_dict(), final)
    print(f"Training complete. Final weights: {final}")


def main():
    parser = argparse.ArgumentParser(description="Train residual CNN for Chrome card artifact removal")
    parser.add_argument("--clean_dir", default="./clean_cards", help="Folder of clean card images")
    parser.add_argument("--output_dir", default="./checkpoints", help="Where to save weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic scratches")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patch_size < 32 or not 0 < args.lr < 1:
        parser.error('epochs/batch must be positive, patch >=32, and learning rate in (0,1)')
    train(args)


if __name__ == "__main__":
    main()
