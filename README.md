# Plant Health AI - Apple Disease Detection

## Project Overview
This project aims to build an AI system for real-time plant health monitoring, focusing on apple diseases. The system uses computer vision and deep learning (CNNs) to classify leaf/stem images and predict disease presence.

## Current Progress
- Dataset downloaded and organized: `APPLE_DISEASE_DATASET`
- Dataset exploration script created (`explore_dataset.py`)
- Initial visualization of plant disease symptoms (spots, color, shape)
- Project structure set up

## Dataset
- Name: Kashmiri Apple Plant Disease Dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/hsmcaju/apple_disease_dataset)
- Note: Dataset not included in repo due to size. Download and place in `data/APPLE_DISEASE_DATASET/`.

## Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
