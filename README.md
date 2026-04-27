# Plant Health AI — Apple Disease Detection System

An end-to-end AI system for real-time plant disease detection using 
computer vision and deep learning. Built with PyTorch, FastAPI, 
and Streamlit.

## What It Does

Classifies apple leaf images into healthy or diseased categories 
using a fine-tuned ResNet-18 CNN model. Exposes a REST API for 
model inference and includes a Streamlit dashboard for real-time 
predictions. Includes latency benchmarking and full model evaluation 
with confusion matrix and per-class F1 reporting.

## Architecture
```User → Streamlit Dashboard → FastAPI Backend → ResNet-18 Model → Prediction```
## Project Structure
src/
- train_pt.py          # Model training with PyTorch
- eval_pt.py           # Evaluation — confusion matrix, F1 reports
- infer_pt.py          # Inference logic
- api_pt.py            # FastAPI backend
- benchmark_latency.py # API latency benchmarking
- pt_data_loader.py    # Dataset loading and augmentation
- explore_dataset.py   # Dataset exploration and visualisation
## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch, ResNet-18, HSV filtering, class balancing |
| API | FastAPI |
| Frontend | Streamlit |
| Evaluation | Confusion matrix, per-class F1, latency benchmarks |
| Environment | Python venv, Git |

## Dataset

- **Name:** Kashmiri Apple Plant Disease Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/hsmcaju/apple_disease_dataset)
- **Note:** Dataset not included due to size. Download and place 
in `data/APPLE_DISEASE_DATASET/`

## How to Run

```bash
# Clone the repo
git clone https://github.com/Chuzzychukwuma/Plant_health_ai.git
cd Plant_health_ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train_pt.py

# Start the API
uvicorn src.api_pt:app --reload

# Run the dashboard
streamlit run src/infer_pt.py

# Run latency benchmarks
python src/benchmark_latency.py
```

## Key Results

| Metric | Result |
|---|---|
| Validation Accuracy | 72.6% |
| API Latency (mean) | ~42–53ms depending on input type |
| p95 Latency | under 58ms |

## Key Features

- Fine-tuned ResNet-18 with HSV filtering, data augmentation, and 
class balancing for robust disease classification
- Validation accuracy of 72.6% on the Kashmiri Apple Disease Dataset
- FastAPI backend with optimised CPU inference — mean latency of 
~42ms on real validation images, ~53ms on random inputs
- p95 latency under 58ms, suitable for real-time prediction use cases
- Full evaluation pipeline — confusion matrix, per-class F1 scores, 
saved reports
- Latency benchmarking comparing process_ms vs total_ms response times
- Modular codebase separating training, inference, API, and evaluation

## What This Taught Me About Cloud Deployment

This project was built with cloud deployment in mind. The FastAPI backend follows the same patterns used when deploying ML models on AWS (SageMaker endpoints, Lambda functions). Containerising this with Docker and deploying to AWS EC2 or ECS is the natural next step which I'm working toward as part of my cloud engineering journey.

## Author

Johnpaul Chukwuma — [LinkedIn](https://www.linkedin.com/in/johnpaul-chukwuma-10557933a) 
| [Medium](https://medium.com/@johnpaulcchukwuma)
