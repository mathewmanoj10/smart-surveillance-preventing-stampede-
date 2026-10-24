"""
finetune_csrnet.py

Fine-tune CSRNet on your own annotated frames.

Input : federated/crowd_yolo_dataset/{images,labels}/{train,val}  (YOLO boxes)
Output: weights/csrnet_finetuned.pth  (best validation MAE)

Boxes are converted to approximate head points, then to Gaussian density maps.
Run from the project root:  python finetune_csrnet.py
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.csrnet import CSRNet, load_csrnet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = os.path.join("federated", "crowd_yolo_dataset")
BASE_WEIGHTS = os.path.join("weights", "PartAmodel_best.pth.tar")
OUT_WEIGHTS = os.path.join("weights", "csrnet_finetuned.pth")

MAX_SIDE = 512          # use the SAME value as MAX_SIDE in main.py
SIGMA = 4.0             # Gaussian spread of one head (input pixels)
HEAD_OFFSET = 0.25      # head point = box centre moved up by 25% of box height
                        # (set to 0.0 for dense frames where boxes are heads)

EPOCHS = 30
FREEZE_FRONTEND_EPOCHS = 10   # train only back-end first
LR = 1e-5
COUNT_LOSS_WEIGHT = 1e-3
SEED = 42

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# DATASET
# ============================================================

class CrowdDataset(Dataset):

    def __init__(self, split, train):
        self.train = train
        img_dir = os.path.join(DATASET, "images", split)
        self.lbl_dir = os.path.join(DATASET, "labels", split)

        patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        self.images = sorted(
            p for pat in patterns
            for p in glob.glob(os.path.join(img_dir, pat))
        )

    def __len__(self):
        return len(self.images)

    def _points(self, image_path, rw, rh):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.lbl_dir, stem + ".txt")
        points = []

        if not os.path.isfile(label_path):
            return points

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()

                if len(parts) != 5 or int(float(parts[0])) != 0:
                    continue

                _, xc, yc, bw, bh = map(float, parts)
                x = xc * rw
                y = (yc - HEAD_OFFSET * bh) * rh
                points.append((x, y))

        return points

    def __getitem__(self, index):
        path = self.images[index]
        img = cv2.imread(path)

        if img is None:
            raise RuntimeError(f"Could not read {path}")

        h, w = img.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))

        if scale < 1.0:
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        rh, rw = img.shape[:2]
        points = self._points(path, rw, rh)

        # CSRNet downsamples by 8, so crop to a multiple of 8
        h8, w8 = rh // 8, rw // 8
        h, w = h8 * 8, w8 * 8
        img = img[:h, :w]

        gt = np.zeros((h, w), dtype=np.float32)

        for x, y in points:
            xi, yi = int(round(x)), int(round(y))

            if 0 <= xi < w and 0 <= yi < h:
                gt[yi, xi] += 1.0

        gt = cv2.GaussianBlur(gt, (0, 0), SIGMA)

        # Area-average then x64 keeps the total count unchanged
        gt_small = cv2.resize(
            gt, (w8, h8), interpolation=cv2.INTER_AREA
        ) * 64.0

        if self.train:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                gt_small = np.ascontiguousarray(gt_small[:, ::-1])

            img = np.clip(
                img.astype(np.float32) * random.uniform(0.8, 1.2), 0, 255
            )

        rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        x = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        y = torch.from_numpy(gt_small.copy()).unsqueeze(0)

        return x, y


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for x, gt in loader:
            pred = model(x.to(device)).sum().item()
            errors.append(abs(pred - gt.sum().item()))

    if not errors:
        return float("inf"), float("inf")

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return mae, rmse


def set_frontend_trainable(model, trainable):
    for p in model.frontend.parameters():
        p.requires_grad = trainable


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_set = CrowdDataset("train", train=True)
    val_set = CrowdDataset("val", train=False)

    print(f"Train images: {len(train_set)} | Val images: {len(val_set)}")

    if len(train_set) == 0:
        print("No training images found in", DATASET)
        return

    # Batch size 1 because image sizes differ
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    if os.path.isfile(BASE_WEIGHTS):
        model = load_csrnet(BASE_WEIGHTS, device=device)
    else:
        print("[WARNING] Base weights not found; training from scratch.")
        model = CSRNet().to(device)

    eval_loader = val_loader if len(val_set) > 0 else train_loader

    mae, rmse = evaluate(model, eval_loader, device)
    print(f"Before fine-tuning: MAE={mae:.2f}  RMSE={rmse:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    best_mae = mae

    os.makedirs(os.path.dirname(OUT_WEIGHTS), exist_ok=True)

    for epoch in range(EPOCHS):
        set_frontend_trainable(model, epoch >= FREEZE_FRONTEND_EPOCHS)
        model.train()
        total = 0.0

        for x, gt in train_loader:
            x, gt = x.to(device), gt.to(device)
            pred = model(x)

            loss = mse(pred, gt) + COUNT_LOSS_WEIGHT * torch.abs(
                pred.sum() - gt.sum()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        mae, rmse = evaluate(model, eval_loader, device)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"loss={total / len(train_loader):.5f} | "
            f"val MAE={mae:.2f} RMSE={rmse:.2f}"
        )

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), OUT_WEIGHTS)
            print(f"  saved best model -> {OUT_WEIGHTS}")

    print(f"\nDone. Best MAE: {best_mae:.2f}")

    if not os.path.isfile(OUT_WEIGHTS):
        print("No improvement over the base model; nothing was saved.")


if __name__ == "__main__":
    main()