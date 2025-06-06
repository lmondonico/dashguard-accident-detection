# DashGuard: Hierarchical Attention for Dashcam Video Accident Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [DashGuard: Hierarchical Attention for Dashcam Video Accident Detection](#dashguard-hierarchical-attention-for-dashcam-video-accident-detection)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [System Architecture](#system-architecture)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Data Preprocessing](#data-preprocessing)
    - [Training Models](#training-models)
    - [Evaluation](#evaluation)
    - [Failure Mode Analysis](#failure-mode-analysis)
    - [Visualizing Optical Flow](#visualizing-optical-flow)
  - [License](#license)
  - [Directory Structure](#directory-structure)

## Project Overview

DashGuard is a deep learning framework for traffic accident prediction from dashcam videos. It uses a hierarchical attention model with EfficientNet to process multimodal inputs (RGB frames and optical flow), aiming to detect subtle pre-collision cues by focusing on critical temporal windows. The project addresses challenges in early collision prediction to enhance vehicle safety systems.

## Features

* **Hierarchical Attention:** Analyzes temporal patterns at multiple scales.
* **Multimodal Input:** Processes RGB frames and optical flow.
* **Efficient Feature Extraction:** Uses EfficientNet-B3.
* **Crash-Focused Sampling:** Prioritizes frames around crash events.
* **Comprehensive Evaluation:** Includes cross-validation and ablation studies.

## System Architecture

DashGuard processes dashcam video by:
1.  **Frame Sampling:** Employs a "crash-focused" strategy, sampling 32 frames (300x300 pixels), with 70% concentrated around the event time within a 5-second window.
2.  **Feature Extraction:** EfficientNet-B3 extracts spatial features from RGB frames and 3-channel optical flow images (derived using Farneback algorithm).
3.  **Multimodal Fusion:** RGB and padded optical flow feature vectors (e.g., 2048-dim each) are concatenated for each timestep, resulting in a combined feature vector (e.g., 4096-dim).
4.  **Hierarchical Temporal Transformer:** This novel transformer processes the sequence of combined features at local (all frames) and global (downsampled sequence) temporal resolutions. Their outputs are fused and fed to a classifier to predict accident probability. The model uses Binary Cross-Entropy (BCE) loss.

## Dataset

* **NEXAR Dashcam Collision Prediction Dataset:** Contains 1,500 training videos with binary collision labels and temporal metadata.
* **Data Split:**
    * Training/Validation: 90% (1,350 samples) for 5-fold cross-validation (80/20 split per fold).
    * Test Set: 10% (150 samples) for final evaluation.
    * Stratified sampling maintains class balance.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/DashGuard.git](https://github.com/your-username/DashGuard.git)
    cd DashGuard
    ```
2.  **Create a Python environment:**
    ```bash
    conda create -n dashguard python=3.8
    conda activate dashguard
    ```
3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio numpy pandas opencv-python scikit-learn matplotlib seaborn tqdm timm joblib
    ```
4.  **Download the Dataset:**
    * Obtain the NEXAR Dataset and place it in `./data-nexar/` as per script expectations.

## Usage

### Data Preprocessing

* Scripts like `./utils/optical-flow-feature-extractor.py` and classes like `MultimodalFeatureExtractor` in `./models/dashguard_crossvalidation.py` handle frame extraction (uniform for flow, crash-focused for RGB), optical flow calculation, and feature extraction using CNNs.
* Features are saved as `.npy` files in directories like `features/optical_flow/`, `features/ablation_cache/`, etc.
    Ensure dataset paths and parameters (`NUM_FRAMES`, `FRAME_SIZE`) are correctly set in the scripts.

### Training Models

1.  **Baseline Model:**
    * Script: `./models/baseline.py` (InceptionV3 features, FC classifier, 5-fold CV).
    * Run: `python ./models/baseline.py`
    * Outputs: `results/baseline_cv/`.

2.  **DashGuard (Hierarchical Transformer) with Cross-Validation:**
    * Script: `./models/dashguard_crossvalidation.py` (EfficientNet-B3, multimodal features, Hierarchical Transformer, 5-fold CV).
    * Run: `python ./models/dashguard_crossvalidation.py`
    * Outputs: `results/cv_efficientnet_multimodal/`, feature directories.

3.  **Ablation Studies:**
    * Script: `./models/ablation_study.py` (various model configurations).
    * Run: `python ./models/ablation_study.py`
    * Outputs: `results/ablation_studies/`.

### Evaluation

* Metrics (ROC-AUC, accuracy, etc.) are calculated by training scripts using `./utils/calculate_metrics.py`.
* Confusion matrices and training curves are saved by the scripts.

### Failure Mode Analysis

* Script: `./models/failure_mode_analysis.py` (identifies FNs and FPs for a specific model).
* Run: `python ./models/failure_mode_analysis.py`
* Outputs: Video IDs in `results/video_id_check/`.

### Visualizing Optical Flow

* Script: `./utils/visualize-optical-flow.py`
* Modify `VIDEO_ID_TO_VISUALIZE` and run: `python ./utils/visualize-optical-flow.py`


## License

This project is licensed under the MIT License.

## Directory Structure

```
├── DashGuard_ProjectReport.pdf
├── README.md
├── .gitignore
├── models/
│   ├── ablation_study.py
│   ├── baseline.py
│   ├── dashguard_crossvalidation.py
│   ├── failure_mode_analysis.py
│   └── module_hierarchical_transformer.py
├── utils/
│   ├── calculate_metrics.py
│   ├── data_loader.py
│   ├── optical-flow-feature-extractor.py
│   └── visualize-optical-flow.py
```