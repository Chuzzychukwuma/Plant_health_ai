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

Add classical CV baseline using HSV + LBP for disease detection


## Classical Computer Vision (CCV) Baseline
 
Add classical CV baseline using HSV + LBP for disease detection.
### Details
- Implemented **HSV color thresholding** to detect leaf discoloration and disease regions.  
- Applied **Local Binary Pattern (LBP)** texture analysis to capture surface texture variations.  
- Established a **non-AI baseline** for comparison with the upcoming CNN deep learning model.  
- Prepared **scripts and visual outputs** for testing and demonstration on dataset samples.

This baseline helps validate that disease detection can be performed with traditional computer vision methods before moving to deep learning models.
This code runs the script: python3 classical_cv/hsv_lbp_detect.py


