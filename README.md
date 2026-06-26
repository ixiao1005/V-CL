# Video-based Cognitive Load Assessment Using MediaPipe

This repository provides the code used to extract video-based behavioral features with MediaPipe and evaluate machine-learning baselines for cognitive load classification.


## Computational Environment

The experiments reported in the manuscript were conducted using the following environment:

| Software         |                  Version |
| ---------------- | -----------------------: |
| Python           |                   3.8.18 |
| NumPy            |                   1.24.4 |
| pandas           |                    1.5.1 |
| OpenCV-Python    |                 4.7.0.72 |
| SciPy            |                   1.10.1 |
| MediaPipe        |                  0.10.11 |
| scikit-learn     |                    1.2.1 |
| XGBoost          |                    2.0.3 |
| openpyxl         |                   3.0.10 |
| tqdm             |                   4.65.2 |

The MediaPipe feature-extraction and traditional machine-learning experiments can be executed on a CPU. A GPU is not required.

The tested dependencies are listed in `requirements.txt`.

---

## Project Structure

```text
V-CL/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── videos/
|   |     └── readme.md
│   ├── frame_features/
│   ├── questionnaire/
│   │   └── questionnaire.xlsx
│   ├── video_features.csv
│   └── labels.csv
│
├── scripts/
│   ├── extract_features.py
│   ├── aggregate_features.py
│   ├── build_labels.py
│   └── train_classifiers.py
│
└── outputs/
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ixiao1005/V-CL.git
cd V-CL
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Access

The raw videos contain identifiable facial information and are therefore not included in this public GitHub repository.

The videos are available through the restricted-access dataset repository described in the accompanying manuscript.

After obtaining authorized access, place the video files in:

```text
data/videos/
```

The code and the derived non-identifying files can be used independently of the raw videos when only the baseline classification results need to be reproduced.

---

## Input Files

### Video files

Place authorized video files in:

```text
data/videos/
```

Supported extensions:

```text
.mp4
.avi
.mov
.mkv
.flv
.wmv
```

Video filenames must contain the participant identifier used in the label file.

Example:

```text
P001.mp4
P002.mp4
P003.mp4
```

### Questionnaire file

For authorized local label generation, place the cleaned questionnaire file in:

```text
data/questionnaire/questionnaire.xlsx
```

Expected columns:

```text
participant_id,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12
```

Example:

```csv
participant_id,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12
P001,5,6,5,7,3,4,3,2,7,6,8,7
P002,8,8,7,9,6,7,6,7,4,5,4,5
```

The questionnaire file used by the code must already have passed the data-quality screening described in the manuscript.

### Label file

The generated label file is stored as:

```text
data/labels.csv
```

Expected format:

```csv
participant_id,ICL_label,ECL_label,GCL_label
P001,0,0,1
P002,2,2,2
```

Label encoding:

```text
0 = low
1 = middle
2 = high
```

### Video-level feature file

The aggregated feature table is stored as:

```text
data/video_features.csv
```

Each row represents one video or participant. The identifier column must match the `participant_id` column in `labels.csv`.

---

## Reproducing the Baseline Results

The baseline classification results can be reproduced directly from:

```text
data/video_features.csv
data/labels.csv
```

This procedure does not require access to the raw videos.

### ICL classification

```bash
python scripts/train_classifiers.py \
    --features_path data/video_features.csv \
    --labels_path data/labels.csv \
    --label_col ICL_label \
    --output_dir outputs
```

### ECL classification

```bash
python scripts/train_classifiers.py \
    --features_path data/video_features.csv \
    --labels_path data/labels.csv \
    --label_col ECL_label \
    --output_dir outputs
```

### GCL classification

```bash
python scripts/train_classifiers.py \
    --features_path data/video_features.csv \
    --labels_path data/labels.csv \
    --label_col GCL_label \
    --output_dir outputs
```

---

## Running the Complete Pipeline

The complete pipeline requires authorized access to the raw videos.

### Step 1: Extract frame-level features

```bash
python scripts/extract_features.py \
    --video_folder data/videos \
    --jsonl_dir data/frame_features
```

Output:

```text
data/frame_features/*.jsonl
```

The extraction script applies MediaPipe Face Mesh, Pose, and Hands to each video frame.

The main MediaPipe settings used in the experiments are:

```text
Face Mesh:
    static_image_mode=False
    max_num_faces=1
    refine_landmarks=True

Pose:
    static_image_mode=False
    model_complexity=1
    enable_segmentation=False

Hands:
    static_image_mode=False
    max_num_hands=2
```

Unless explicitly specified in the script, the default MediaPipe detection and tracking confidence thresholds are used.

### Step 2: Aggregate frame-level features

```bash
python scripts/aggregate_features.py \
    --jsonl_dir data/frame_features \
    --out_csv data/video_features.csv
```

Output:

```text
data/video_features.csv
```

The aggregation script converts frame-level measurements into video-level statistical features.

### Step 3: Generate cognitive load labels

```bash
python scripts/build_labels.py \
    --input_path data/questionnaire/questionnaire.xlsx \
    --output_path data/labels.csv
```

Output:

```text
data/labels.csv
```

ICL, ECL, and GCL labels are generated independently.

The script uses the original questionnaire item directions:

```text
ICL: Q1–Q4
ECL: Q5–Q8
GCL: Q9–Q12
```

The GCL items are not reverse scored.

### Step 4: Train and evaluate classifiers

Run the classification script separately for `ICL_label`, `ECL_label`, and `GCL_label`.

Example:

```bash
python scripts/train_classifiers.py \
    --features_path data/video_features.csv \
    --labels_path data/labels.csv \
    --label_col ICL_label \
    --output_dir outputs
```

---

## Implemented Models

The following classifiers are included:

* Support Vector Machine;
* Decision Tree;
* Random Forest;
* XGBoost.

The evaluation procedure uses:

* macro precision;
* macro recall;
* macro F1-score;


---

## Extracted Features

The frame-level extraction script calculates features related to:

* eye openness;
* mouth openness;
* eyebrow distance;
* head pose;
* upper-body pose;
* wrist-to-face distance;
* hand movement;
* body movement;
* MediaPipe detection validity.

The detailed definitions and calculations are provided in the accompanying manuscript.

The aggregation script calculates statistics including:

```text
mean
standard deviation
minimum
maximum
range
10th percentile
25th percentile
50th percentile
75th percentile
90th percentile
mean first-order difference
standard deviation of first-order differences
```

Example aggregated columns:

```text
mp_ear_mean
mp_ear_std
mp_ear_p50
mp_ear_diff_mean
mp_ear_diff_std
```

---

## Output Files

Each classification run generates files such as:

```text
outputs/
├── ICL_label_metrics.csv
├── ICL_label_confusion_matrices.csv
└── ICL_label_results.xlsx
```

The corresponding filenames are generated for ECL and GCL when the value of `--label_col` is changed.

---

## License

The source code is released under the license provided in the `LICENSE` file.

The software license applies only to the code and does not grant access to or permission to redistribute the restricted video data.
