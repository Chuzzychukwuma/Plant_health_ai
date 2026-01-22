# 🍎 Plant Health AI - Apple Disease Detection

## 🌿 Project Overview
This project aims to build a complete AI system for **real-time plant health monitoring**, focusing on **apple leaf diseases**.  
The system uses **computer vision** and **deep learning (CNNs with PyTorch)** to classify images of apple leaves and predict disease presence.  

It includes:
- A **classical computer vision (HSV + LBP)** baseline  
- A **deep learning model (ResNet-18)** trained on the same dataset  
- A **FastAPI backend** for inference  
- A **Streamlit web app** for interactive testing and visualization  

---

## ✅ Current Progress
- Dataset downloaded and organized under `data/APPLE_DISEASE_DATASET`
- Classical computer vision baseline implemented (`classical_cv/hsv_lbp_detect.py`)
- Deep learning pipeline implemented (`src/train_pt.py`, `src/eval_pt.py`, `src/infer_pt.py`)
- Model trained with **72.6% validation accuracy**
- API created with **FastAPI** (`src/api_pt.py`)
- Web interface built with **Streamlit** (`app/streamlit_app.py`)

---

## 📁 Dataset
- **Name:** Kashmiri Apple Plant Disease Dataset  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/hsmcaju/apple_disease_dataset)  
- **Classes:**  
  - Apple Rot Leaves  
  - Healthy Leaves  
  - Leaf Blotch  
  - Scab Leaves  

### Folder Structure
data/
└── APPLE_DISEASE_DATASET/
├── APPLE ROT LEAVES/
├── HEALTHY LEAVES/
├── LEAF BLOTCH/
└── SCAB LEAVES/

> ⚠️ The dataset is not included in the repository due to size limits.  
> Please download it from Kaggle and place it in `data/APPLE_DISEASE_DATASET/`.

---

## ⚙️ Installation & Setup

1. Clone the Repository
```bash
git clone https://github.com/Chuzzychukwuma/Plant_health_ai.git
cd Plant_health_ai

2. Create & Activate Environment

Using Conda (recommended):
conda create -n pt-env python=3.11 -y
conda activate pt-env

3. Install Dependencies
pip install -r requirements.txt

If you don't have a requirements.txt, install manually:
pip install torch torchvision fastapi uvicorn streamlit pandas matplotlib pillow scikit-learn python-multipart


🧩 Classical Computer Vision (Baseline)

Goal: Provide a non-AI baseline for apple leaf disease detection using traditional image processing.

Details

HSV color thresholding detects discolored regions.

Local Binary Pattern (LBP) captures leaf texture variations.

Contours & masks highlight disease-affected regions.

Results are displayed with OpenCV.

Run the Script
python3 classical_cv/hsv_lbp_detect.py
This script outputs segmented disease regions for visual inspection.
It helps compare classical CV vs deep learning performance.

🤖 Deep Learning Model (PyTorch)
Architecture

The model uses ResNet-18 pretrained on ImageNet, fine-tuned on the apple leaf dataset.

Files

src/train_pt.py → trains the model

src/eval_pt.py → evaluates trained models

src/infer_pt.py → predicts a single image

🔹 Train the Model
python -m src.train_pt

This script:

Loads and splits the dataset (80/20 train/val)

Trains a ResNet-18 model in two stages:

Head training – train classifier only

Fine-tuning – unfreeze last block for better feature extraction

Saves:

models/pt_resnet18_best.pth (best checkpoint)

models/pt_resnet18_final.pth (final model)

Accuracy/loss graphs in reports/figures/

Classes: ['APPLE ROT LEAVES', 'HEALTHY LEAVES', 'LEAF BLOTCH', 'SCAB LEAVES']
[Head] Epoch 1/8 | tr_loss 1.4172 acc 0.3284 | val_acc 0.4167
...
[FT] Epoch 10/10 | tr_loss 0.2706 acc 1.0000 | val_acc 0.7262
✅ Saved best model (val_acc=0.7262)

Evaluate the Model
python -m src.eval_pt

This script prints precision, recall, F1-score, and accuracy, and saves:

reports/figures/confusion_matrix_pt.png


Sample Output:

precision  recall  f1-score  support
APPLE ROT LEAVES   0.73   0.67   0.70
HEALTHY LEAVES     0.58   0.78   0.67
LEAF BLOTCH        0.77   0.77   0.77
SCAB LEAVES        0.74   0.72   0.73
Overall Accuracy: 0.7262

🔹 Inference (Test a Single Image)
python -m src.infer_pt data/APPLE_DISEASE_DATASET/HEALTHY\ LEAVES/1001.jpg.jpeg


Output Example:

Predicted: HEALTHY LEAVES (93.5% confidence)

⚡ FastAPI Backend (REST API)

The FastAPI server lets you send images via HTTP and receive predictions.

Run the API
python -m uvicorn src.api_pt:app --reload --port 8000

Endpoints
Method	Route	Description
GET	/health	Check model status and classes
POST	/predict	Upload image to get prediction

Example:

curl http://127.0.0.1:8000/health


Output:

{"status":"ok","device":"cpu","classes":["APPLE ROT LEAVES","HEALTHY LEAVES","LEAF BLOTCH","SCAB LEAVES"]}


Open in browser for Swagger UI:
👉 http://127.0.0.1:8000/docs

💻 Streamlit Web App

The Streamlit frontend provides an interactive dashboard for uploading and testing leaf images visually.

Run Streamlit
streamlit run app/streamlit_app.py

Features

Upload any apple leaf image

See predicted disease and confidence

Display processed image preview

Runs on localhost:8501

📊 Results Summary
Metric	Value
Best Validation Accuracy	72.6%
Model Architecture	ResNet-18 (fine-tuned)
Training Data Split	80% train / 20% validation
Evaluation Metrics	Precision, Recall, F1-Score
Output Graphs	Saved under reports/figures/

Project Structure
Plant_health_ai/
├── app/
│   └── streamlit_app.py
├── classical_cv/
│   └── hsv_lbp_detect.py
├── data/
│   └── APPLE_DISEASE_DATASET/
├── models/
│   ├── pt_resnet18_best.pth
│   └── pt_resnet18_final.pth
├── reports/
│   └── figures/
├── src/
│   ├── api_pt.py
│   ├── eval_pt.py
│   ├── infer_pt.py
│   ├── pt_data_loader.py
│   └── train_pt.py
├── requirements.txt
└── README.md
