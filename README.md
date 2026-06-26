# Video-based Cognitive Load Assessment using MediaPipe

This repository provides a complete pipeline for cognitive load assesment from videos using structured features extracted with MediaPipe and machine learning models.
---
## Pipeline

The code implements the following workflow:

Video files
    ↓
Frame-level MediaPipe feature extraction
    ↓
Video-level statistical aggregation
    ↓
Questionnaire-derived label generation
    ↓
Five-fold baseline classification

The classification tasks are performed separately for:

ICL: Intrinsic Cognitive Load
ECL: Extraneous Cognitive Load
GCL: Germane Cognitive Load
---
## Computational Environment

The experiments reported in the accompanying manuscript were conducted
using the following environment:

- Operating system: Ubuntu 20.04.5 LTS
- Python: 3.8.18
- NumPy: 1.24.4
- pandas: 1.5.1
- OpenCV-Python: 4.7.0.72
- SciPy: 1.10.1
- MediaPipe: 0.10.11
- scikit-learn: [actual version]
- XGBoost: [actual version]
- openpyxl: [actual version]

The MediaPipe feature-extraction and traditional machine-learning
experiments were executed on the CPU. A GPU is not required.

All Python dependencies and their tested versions are listed in
`requirements.txt`.
---


## 📌 Overview

This project provides a reproducible pipeline for video-based cognitive load assessment.

We extract structured features from:

- Facial
- Body Pose
- Motion

and perform classification using machine learning models.

---

## 🗂️ Project Structure

```text
V-CL/
├── data/
│   ├── videos/
│   ├── questionnaire/
│   │   └── questionnaire.xlsx
│   └── labels.csv
├── scripts/
│   ├── extract_features.py
│   ├── aggregate_features.py
│   ├── build_labels.py
│   └── train_classifiers.py
├── outputs/
└── README.md
```

---

## ⚙️ Installation
### 1. Clone and enter repo:
```bash 
git clone https://github.com/ixiao1005/V-CL.git
cd V-CL
```

### 2.Create a virtualenv (optional but recommended):
```
python3 -m venv .venv
source .venv/bin/activate
```

### 3.Install dependencies:
```
pip install -r requirements.txt
```
---

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

The repository have two versions of questionnaire data:

####  Questionnaire Data
```
data/questionnaire/questionnaire.xlsx
```

This file contains the  responses collected from participants.



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

---

## 🚀 Pipeline
### Step 1 — Extract MediaPipe Features
```bash
python scripts/extract_features.py \
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
    --input_path data/questionnaire/questionnaire.xlsx \
    --output_path data/labels.csv
```
### Step 4 — Train Classifiers

The model can be trained separately for each type of cognitive load:

- **ICL** — Intrinsic Cognitive Load  
- **ECL** — Extraneous Cognitive Load  
- **GCL** — Germane Cognitive Load  

This is controlled by the `--label_col` argument.



#### Train for ICL

```bash
python scripts/train_classifiers.py \
    --features_path data/video_features.csv \
    --labels_path data/labels.csv \
    --label_col ICL_label \
    --output_dir outputs
```
#### Train for ECL

```bash
python scripts/train_classifiers.py \
    --features_path data/video_features.csv \
    --labels_path data/labels.csv \
    --label_col ECL_label \
    --output_dir outputs
```
#### Train for GCL

```bash
python scripts/train_classifiers.py \
    --features_path data/video_features.csv \
    --labels_path data/labels.csv \
    --label_col GCL_label \
    --output_dir outputs
```
#### Outputs
Each run will generate:
```text
outputs/
├── ICL_label_metrics.csv
├── ICL_label_confusion_matrices.csv
├── ICL_label_results.xlsx
```

(similarly for ECL and GCL)

---

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

---

## 🧠 Feature Description
Detailed definitions of the features are provided in the paper. Here, we present a concise summary. All features are extracted using MediaPipe and aggregated from frame-level measurements into video-level statistical representations.


| Category | Feature | Description |
|----------|--------|-------------|
| **Facial** | EAR (Eye Aspect Ratio) | Measures eye openness; used to detect blinking or eye closure |
|  | MAR (Mouth Aspect Ratio) | Measures mouth openness |
|  | Frown distance | Distance between eyebrows, indicating frowning intensity |
|  | Head pitch | Vertical head rotation angle |
|  | Head yaw | Horizontal head rotation angle |
|  | Head roll | Head tilt angle |

---

| Category | Feature | Description |
|----------|--------|-------------|
| **Body Pose** | Torso pitch | Forward/backward body inclination |
|  | Torso roll | Left/right body tilt |
|  | Shoulder distance | Distance between left and right shoulders |
|  | Left wrist to face | Distance from left wrist to face (nose) |
|  | Right wrist to face | Distance from right wrist to face (nose) |

---

| Category | Feature | Description |
|----------|--------|-------------|
| **Motion** | Hand motion | Frame-to-frame movement of wrists |
|  | Body motion | Frame difference intensity |

---


### Feature Aggregation

All frame-level features are aggregated into video-level statistics, including:

- Mean, standard deviation, minimum, maximum
- Range
- Percentiles (10%, 25%, 50%, 75%, 90%)
- First-order difference statistics (temporal dynamics)

Example:
- mp_ear_mean, mp_ear_std, mp_ear_p50, mp_ear_diff_std

These aggregated features are used as input for machine learning models.

