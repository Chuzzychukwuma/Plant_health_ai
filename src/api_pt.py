# src/api_pt.py
import io
import time
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from src.pt_data_loader import get_loaders_stratified

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Plant Health AI", version="1.0")

# ---------------------------------------
# ✅ Allow Streamlit / local frontend CORS
# ---------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# 🔹 Load model + class names once at startup
# ---------------------------------------


@app.on_event("startup")
def load_model():
    global MODEL, CLASS_NAMES, TF
    _, _, CLASS_NAMES = get_loaders_stratified(
        dataset_path="data/APPLE_DISEASE_DATASET",
        img_size=224, batch_size=16, val_split=0.2, seed=42
    )

    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(CLASS_NAMES))
    m.load_state_dict(torch.load(
        "models/pt_resnet18_best.pth", map_location=DEVICE))
    m.to(DEVICE).eval()
    MODEL = m

    TF = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ---------------------------------------
# 🔹 Health check endpoint
# ---------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": CLASS_NAMES}

# ---------------------------------------
# 🔹 Prediction endpoint (with latency timing)
# ---------------------------------------


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t_start = time.perf_counter()  # total request start

    # --- Image decoding ---
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # --- Preprocessing ---
    x = TF(img).unsqueeze(0).to(DEVICE)

    # --- Model inference (timed) ---
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = MODEL(x)
    t1 = time.perf_counter()

    # --- Post-processing ---
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = int(probs.argmax())

    process_ms = (t1 - t0) * 1000.0        # model forward pass only
    total_ms = (time.perf_counter() - t_start) * 1000.0  # full request time

    return {
        "prediction": CLASS_NAMES[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "probs": {cls: round(float(p), 6) for cls, p in zip(CLASS_NAMES, probs)},
        "process_ms": round(process_ms, 2),  # forward pass time (ms)
        "total_ms": round(total_ms, 2)       # full API latency (ms)
    }
