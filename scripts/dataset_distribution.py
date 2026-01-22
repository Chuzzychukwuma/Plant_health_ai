# scripts/dataset_distribution.py
import pandas as pd
from collections import Counter
from src.pt_data_loader import get_loaders_stratified


def count_labels(loader):
    labels = []
    for _, y in loader:
        labels.extend(y.tolist())
    return Counter(labels)


def main():
    train_loader, val_loader, class_names = get_loaders_stratified(
        dataset_path="data/APPLE_DISEASE_DATASET",
        img_size=224,
        batch_size=16,
        val_split=0.2,
        seed=42
    )

    train_counts = count_labels(train_loader)
    val_counts = count_labels(val_loader)

    rows = []
    for idx, cls in enumerate(class_names):
        rows.append({
            "Class": cls,
            "Training Images": train_counts.get(idx, 0),
            "Validation Images": val_counts.get(idx, 0),
            "Total": train_counts.get(idx, 0) + val_counts.get(idx, 0)
        })

    df = pd.DataFrame(rows)
    print(df)

    # Save for report
    df.to_csv("reports/metrics/dataset_distribution.csv", index=False)
    print("Saved to reports/metrics/dataset_distribution.csv")


if __name__ == "__main__":
    main()
