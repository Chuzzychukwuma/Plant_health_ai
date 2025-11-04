# src/pt_data_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split
import torch

# --- Default transforms (good baseline) ---


def _build_transforms(img_size: int):
    tf_train = transforms.Compose([
        # stronger aug helps generalization
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return tf_train, tf_val


def get_loaders(dataset_path='data/APPLE_DISEASE_DATASET',
                img_size=224, batch_size=16, val_split=0.2, seed=42):
    """Simple random split (non-stratified). Kept for reference."""
    tf_train, tf_val = _build_transforms(img_size)

    full = datasets.ImageFolder(root=dataset_path, transform=tf_train)
    class_names = full.classes

    total = len(full)
    val_len = int(total * val_split)
    train_len = total - val_len
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    train_ds, val_ds = random_split(full, [train_len, val_len], generator=gen)

    # ensure val uses val transforms
    val_ds.dataset.transform = tf_val

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, class_names


def get_loaders_stratified(dataset_path='data/APPLE_DISEASE_DATASET',
                           img_size=224, batch_size=16, val_split=0.2, seed=42):
    """Stratified split so each class keeps its proportion in train/val."""
    tf_train, tf_val = _build_transforms(img_size)

    full = datasets.ImageFolder(root=dataset_path, transform=tf_train)
    class_names = full.classes

    # labels for stratification
    y = [label for _, label in full.samples]

    train_idx, val_idx = train_test_split(
        range(len(y)),
        test_size=val_split,
        random_state=seed,
        stratify=y
    )

    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx)
    # ensure val uses val transforms
    val_ds.dataset.transform = tf_val

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, class_names


def get_class_weights_from_indices(dataset, indices, num_classes):
    """Optional: compute class weights for imbalanced data."""
    import numpy as np
    labels = [dataset.samples[i][1] for i in indices]
    counts = np.bincount(labels, minlength=num_classes)
    # inverse frequency as weight (avoid div by zero)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)
