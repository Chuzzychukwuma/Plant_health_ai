# src/eval_pt.py
import os
import numpy as np
import torch
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from src.pt_data_loader import get_loaders_stratified

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, num_classes):
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m = m.to(DEVICE)
    m.eval()
    return m


def main():
    _, val_loader, class_names = get_loaders_stratified(
        dataset_path='data/APPLE_DISEASE_DATASET',
        img_size=224,
        batch_size=16,
        val_split=0.2,
        seed=42
    )
    model_path = "models/pt_resnet18_best.pth"
    if not os.path.exists(model_path):
        raise SystemExit("Train first: python -m src.train_pt")

    model = load_model(model_path, len(class_names))

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE)
            out = model(X)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())

    print(classification_report(y_true, y_pred,
          target_names=class_names, digits=4))
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right')
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs("reports/figures", exist_ok=True)
    plt.tight_layout()
    out_path = "reports/figures/confusion_matrix_pt.png"
    plt.savefig(out_path)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
