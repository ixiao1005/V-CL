# Video-based Cognitive Load Classification with MediaPipe Features

This repository provides a complete pipeline for cognitive load classification from videos using MediaPipe features and machine learning models.

The pipeline includes:
- Frame-level feature extraction using MediaPipe
- Aggregation into video-level features
- Label generation from questionnaire data
- Classification using machine learning models

---

## 📌 Overview

This project focuses on **lightweight and interpretable video-based cognitive load estimation** using structured features derived from:

- Facial expressions
- Body pose
- Hand movements

Compared to deep learning approaches, this method:
- Requires less data
- Is computationally efficient
- Provides interpretable features

---

## 🗂️ Project Structure

```text
V-CL-dataset/
├── README.md
├── requirements.txt
├── data/
│   ├── questionnaire/
│   │   ├── questionnaire_raw.xlsx
│   │   └── questionnaire_reverse_scored.xlsx
│   ├── videos/
│   └── labels.csv
├── scripts/
│   ├── extract_mediapipe_features.py
│   ├── aggregate_features.py
│   ├── build_labels.py
│   └── train_ml_classifiers.py
├── outputs/
```

## ⚙️ Installation
```bash 
git clone https://github.com/yourname/V-CL-dataset.git
cd V-CL-dataset
pip install -r requirements.txt
```

## 📊 Data Format
### 1. Video Data

Place videos in:
```
data/videos/
```
Supported formats:
```
.mp4, .avi, .mov, .mkv, .flv, .wmv
```
### 2. Questionnaire Data
```
data/questionnaire/questionnaire_reverse_scored.xlsx
```
Expected columns:
```
participant_id, Q1 ... Q12
```
### 3. Labels File

The label file is generated from questionnaire data using:

```bash
python scripts/build_labels.py \
    --input_path data/questionnaire/questionnaire_reverse_scored.xlsx \
    --output_path data/labels.csv
```
Output:
```
data/labels.csv
```

Format:
```csv
participant_id,ICL_label,ECL_label,GCL_label
P001,0,0,1
P002,2,2,2
```
## 🚀 Pipeline
### Step 1 — Extract MediaPipe Features
```bash
python scripts/extract_mediapipe_features.py \
    --video_folder data/videos \
    --jsonl_dir data/frame_features
```
Output:
```
data/frame_features/*.jsonl
```
### Step 2 — Aggregate Frame Features
```bash
python scripts/aggregate_features.py \
    --jsonl_dir data/frame_features \
    --out_csv data/video_features.csv
```
### Step 3 — Generate Labels
```bash
python scripts/build_labels.py \
    --input_path data/questionnaire/questionnaire_reverse_scored.xlsx \
    --output_path data/labels.csv
```
### Step 4 — Train Classifiers
```bash
python scripts/train_ml_classifiers.py \
    --features_path data/video_features.csv \
    --labels_path data/labels.csv \
    --label_col ICL_label \
    --output_dir outputs
```

## 🤖 Models

Implemented models:

- SVM 
- Decision Tree
- Random Forest
- XGBoost

Evaluation:

- 5-fold Stratified Cross Validation
- Grid Search hyperparameter tuning

Metrics:

- Macro F1-score
- Recall
- Precision

## 📈 Outputs

Saved in:
```
outputs/
```
Files include:

- ```*_metrics.csv```

- ``` *_confusion_matrices.csv```

- ```*_results.xlsx```

## 🧠 Feature Description
Facial Features

- EAR (Eye Aspect Ratio)

- MAR (Mouth Aspect Ratio)

- Frown distance

- Head pose (pitch, yaw, roll)

Body Features

- Torso pitch & roll

- Shoulder distance

- Wrist-to-face distance

Motion Features

- Hand motion

- Body motion

Behavioral Indicators

- Blink-like rate

- Head turn rate

- Stillness rate

