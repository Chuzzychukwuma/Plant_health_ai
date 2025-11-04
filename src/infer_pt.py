import torch
from torchvision import models, transforms
from PIL import Image
from src.pt_data_loader import get_loaders_stratified

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def predict(img_path):
    _, _, class_names = get_loaders_stratified(
        dataset_path="data/APPLE_DISEASE_DATASET",
        img_size=224, batch_size=16, val_split=0.2, seed=42
    )
    model = load_model("models/pt_resnet18_best.pth", len(class_names))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    conf, cls = torch.max(probs, 0)
    print(f"Predicted: {class_names[cls]} ({conf*100:.1f}% confidence)")
    return class_names[cls], conf.item()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.infer_pt path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])
