# src/pt_data_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch


def get_loaders(
    dataset_path='data/APPLE_DISEASE_DATASET',
    img_size=224,
    batch_size=16,
    val_split=0.2,
    seed=42
):
    """
    Returns: train_loader, val_loader, class_names
    Expects folder structure:
        data/APPLE_DISEASE_DATASET/
            ClassA/
                img1.jpg ...
            ClassB/
                ...
    """

    # --- transforms ---
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # ImageNet stats
    ])

    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- dataset ---
    full = datasets.ImageFolder(root=dataset_path, transform=tf_train)
    class_names = full.classes

    # --- split ---
    total = len(full)
    val_len = int(total * val_split)
    train_len = total - val_len
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    train_ds, val_ds = random_split(full, [train_len, val_len], generator=gen)
    # ensure val uses validation transform
    val_ds.dataset.transform = tf_val

    # --- loaders (macOS-safe: num_workers=0) ---
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    return train_loader, val_loader, class_names


if __name__ == "__main__":
    # quick smoke test
    tr, val, classes = get_loaders()
    X, y = next(iter(tr))
    print("Classes:", classes)
    print("Batch shapes:", X.shape, y.shape)
