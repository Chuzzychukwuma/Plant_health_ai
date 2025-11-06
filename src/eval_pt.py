# src/eval_pt.py
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
from src.pt_data_loader import get_loaders_stratified

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "data/APPLE_DISEASE_DATASET"
MODEL_PATH = "models/pt_resnet18_best.pth"
IMG_SIZE = 224
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42


def load_model(path, num_classes):
    # weights not needed; we'll load checkpoint
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m = m.to(DEVICE)
    m.eval()
    return m


def main():
    # --- data ---
    _, val_loader, class_names = get_loaders_stratified(
        dataset_path=DATASET_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            "Model not found. Train first with: python -m src.train_pt")

    # --- model ---
    model = load_model(MODEL_PATH, len(class_names))

    # --- inference on val set ---
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE)
            logits = model(X)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    # --- metrics ---
    report_txt = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)
    print(report_txt)

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    df_report = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_true, y_pred)

    # --- ensure output dirs ---
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)

    # --- save text / csv reports ---
    with open("reports/metrics/classification_report.txt", "w") as f:
        f.write(report_txt)
    df_report.to_csv("reports/metrics/classification_report.csv", index=True)

    # --- Confusion Matrix (seaborn heatmap) ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - ResNet18 (PyTorch)")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix_pt.png")
    plt.close()
    print("Saved confusion matrix to reports/figures/confusion_matrix_pt.png")

    # --- Per-class F1 bar chart ---
    plt.figure(figsize=(8, 5))
    df_report.iloc[:-3]["f1-score"].plot(kind='bar')
    plt.title("Per-Class F1 Scores - ResNet18 (PyTorch)")
    plt.ylabel("F1-score")
    plt.xlabel("Class")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("reports/figures/f1_scores_pt.png")
    plt.close()
    print("Saved per-class F1 scores to reports/figures/f1_scores_pt.png")


if __name__ == "__main__":
    main()
