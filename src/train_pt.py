# src/train_pt.py
import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score
from src.pt_data_loader import get_loaders_stratified

# ------------------------------
# Reproducibility (as much as possible)
# ------------------------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # True can slow a lot on CPU


set_seed(42)

# ------------------------------
# Config
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 8          # train the new FC head with backbone frozen
FT_EPOCHS = 10           # fine-tune last block briefly
LR_HEAD = 1e-4
LR_FT = 5e-5
WEIGHT_DECAY = 1e-4
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# where we’ll save metrics & figures
Path("reports/metrics").mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# ------------------------------
# Data
# ------------------------------
train_loader, val_loader, class_names = get_loaders_stratified(
    dataset_path="data/APPLE_DISEASE_DATASET",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    val_split=0.2,
    seed=42
)
num_classes = len(class_names)
print("Classes:", class_names)

# ------------------------------
# Model (ResNet18) + replace head
# ------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# freeze all
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# ------------------------------
# Loss, Optim, Scheduler
# ------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=LR_HEAD,
                       weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_EPOCHS)

best_val = 0.0
best_path = os.path.join(MODEL_DIR, "pt_resnet18_best.pth")
final_path = os.path.join(MODEL_DIR, "pt_resnet18_final.pth")

# ------------------------------
# History (for CSV + plots)
# ------------------------------
hist = []  # rows: {"phase","epoch","tr_loss","tr_acc","val_loss","val_acc"}


def train_one_epoch(m, loader, crit, opt):
    m.train()
    losses, preds, trues = [], [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = m(X)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds.extend(out.argmax(1).detach().cpu().numpy())
        trues.extend(y.detach().cpu().numpy())
    return float(np.mean(losses)), accuracy_score(trues, preds)


def eval_one_epoch(m, loader, crit):
    m.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = m(X)
            loss = crit(out, y)
            losses.append(loss.item())
            preds.extend(out.argmax(1).detach().cpu().numpy())
            trues.extend(y.detach().cpu().numpy())
    return float(np.mean(losses)), accuracy_score(trues, preds)


# ------------------------------
# Phase 1: Train head (frozen backbone)
# ------------------------------
for epoch in range(1, HEAD_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(
        model, train_loader, criterion, optimizer)
    va_loss, va_acc = eval_one_epoch(model, val_loader, criterion)
    scheduler.step()
    dt = time.time() - t0
    print(f"[Head] Epoch {epoch}/{HEAD_EPOCHS} | tr_loss {tr_loss:.4f} acc {tr_acc:.4f} "
          f"| val_loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")

    # log metrics
    hist.append({
        "phase": "Head",
        "epoch": epoch,
        "tr_loss": tr_loss,
        "tr_acc": tr_acc,
        "val_loss": va_loss,
        "val_acc": va_acc,
    })

    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), best_path)
        print(f"  ✅ Saved best (val_acc={best_val:.4f})")

# ------------------------------
# Phase 2: Fine-tune last block + head
# ------------------------------
# unfreeze only layer4 + fc
for p in model.parameters():
    p.requires_grad = False
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR_FT, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FT_EPOCHS)

for epoch in range(1, FT_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(
        model, train_loader, criterion, optimizer)
    va_loss, va_acc = eval_one_epoch(model, val_loader, criterion)
    scheduler.step()
    dt = time.time() - t0
    print(f"[FT]   Epoch {epoch}/{FT_EPOCHS}   | tr_loss {tr_loss:.4f} acc {tr_acc:.4f} "
          f"| val_loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")

    # log metrics
    hist.append({
        "phase": "FT",
        "epoch": epoch,
        "tr_loss": tr_loss,
        "tr_acc": tr_acc,
        "val_loss": va_loss,
        "val_acc": va_acc,
    })

    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), best_path)
        print(f"  ✅ Saved best (val_acc={best_val:.4f})")

# save final
torch.save(model.state_dict(), final_path)
print(f"Done. Best val acc: {best_val:.4f}")
print(f"Saved: {best_path} and {final_path}")

# ------------------------------
# Save history CSV + plots
# ------------------------------
df = pd.DataFrame(hist)
df.to_csv("reports/metrics/train_history.csv", index=False)

# Accuracy plot
plt.figure()
for phase in ["Head", "FT"]:
    sub = df[df["phase"] == phase]
    if not sub.empty:
        plt.plot(sub["epoch"], sub["val_acc"], label=f"{phase} val_acc")
        plt.plot(sub["epoch"], sub["tr_acc"],
                 linestyle="--", label=f"{phase} tr_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch (Head & FT)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/accuracy_vs_epoch.png", dpi=150)
plt.close()

# Loss plot
plt.figure()
for phase in ["Head", "FT"]:
    sub = df[df["phase"] == phase]
    if not sub.empty:
        plt.plot(sub["epoch"], sub["val_loss"], label=f"{phase} val_loss")
        plt.plot(sub["epoch"], sub["tr_loss"],
                 linestyle="--", label=f"{phase} tr_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch (Head & FT)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/loss_vs_epoch.png", dpi=150)
plt.close()
