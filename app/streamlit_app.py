import io
import os
import time
import requests
import streamlit as st
from PIL import Image

# ---- Settings ----
DEFAULT_API = "http://127.0.0.1:8000/predict"
API_URL = os.getenv("PLANT_API_URL", DEFAULT_API)

st.set_page_config(page_title="🌿 Plant Health AI",
                   page_icon="🌿", layout="centered")
st.title("🌿 Plant Health AI — Leaf Disease Classifier")
st.caption("Upload an apple leaf image to get the predicted class and confidence (served by your FastAPI model).")

with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("API endpoint", API_URL)
    st.caption(
        "Tip: if you deploy the API elsewhere, paste its URL here (e.g. Render/EC2).")
    st.divider()
    st.caption("Model: ResNet-18 (PyTorch)")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Preview
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Serialize image as JPEG bytes for multipart/form-data
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    if st.button("Predict", type="primary"):
        t0 = time.time()
        with st.spinner("Calling API…"):
            try:
                files = {"file": ("image.jpg", buf, "image/jpeg")}
                r = requests.post(api_url, files=files, timeout=60)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        pred = data.get("prediction", "N/A")
        conf = float(data.get("confidence", 0.0)) * 100
        probs = data.get("probs", {})

        st.success(
            f"Prediction: **{pred}**  \nConfidence: **{conf:.1f}%**  \nLatency: {(time.time()-t0)*1000:.0f} ms")

        if probs:
            st.subheader("Class probabilities")
            for cls, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                pct = float(p) * 100
                st.write(f"{cls}: {pct:.1f}%")
                st.progress(min(max(float(p), 0.0), 1.0))
