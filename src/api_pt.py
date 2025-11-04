# src/api_pt.py
import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms
from src.pt_data_loader import get_loaders_stratified

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Plant Health AI", version="1.0")

# (optional) allow local frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Load model + class names once at startup -----


@app.on_event("startup")
def load_model():
    global model, class_names, tf
    _, _, class_names = get_loaders_stratified(
        dataset_path="data/APPLE_DISEASE_DATASET",
        img_size=224, batch_size=16, val_split=0.2, seed=42
    )
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(class_names))
    m.load_state_dict(torch.load(
        "models/pt_resnet18_best.pth", map_location=DEVICE))
    m.to(DEVICE).eval()
    model = m

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": class_names}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(probs.argmax())
    return {
        "prediction": class_names[top_idx],
        "confidence": float(probs[top_idx]),
        "probs": {cls: float(p) for cls, p in zip(class_names, probs)}
    }
