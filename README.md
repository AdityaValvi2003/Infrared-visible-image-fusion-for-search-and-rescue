# Infrared and Visible Image Fusion for Search and Rescue

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![YOLO](https://img.shields.io/badge/Detector-YOLOv8-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning pipeline that fuses infrared and visible images to improve
object detection in low-light and nighttime environments. Developed as a
B.Tech final year project at COEP Technological University, Pune.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Team](#team)

---

## Overview

Conventional object detection systems rely on visible-spectrum imagery,
which degrades significantly in low-light or nighttime conditions. This
project addresses that limitation by fusing infrared (thermal) and visible
images before passing the result into a YOLO-based detector.

The pipeline consists of four stages:
1. **Image Acquisition** — paired IR and visible images
2. **Preprocessing** — resize, normalize, align, CLAHE enhancement
3. **Fusion** — thermal cues from IR + texture from visible
4. **Detection** — YOLOv8 with confidence thresholding and NMS

---

## Features

- Paired infrared and visible image fusion
- CLAHE-based low-light contrast enhancement
- YOLO-based object detection on fused output
- Comparative evaluation against visible-only and IR-only baselines
- Supports person, car, motorcycle, bicycle detection in nighttime scenes

---

## Project Structure
├── data/
│   ├── infrared/          # IR image pairs
│   ├── visible/           # Visible image pairs
│   └── fused/             # Fusion outputs
├── preprocessing/
│   └── preprocess.py      # Resize, normalize, CLAHE, align
├── fusion/
│   └── fuse.py            # Fusion algorithm
├── detection/
│   └── detect.py          # YOLOv8 inference + NMS
├── results/
│   ├── quantitative/      # Precision, Recall, mAP tables
│   └── qualitative/       # Output detection images
├── requirements.txt
└── README.md

---

## Dataset

The project uses paired infrared and visible image datasets captured under
challenging low-light and nighttime conditions.

| Subset     | Samples | Purpose               |
|------------|---------|-----------------------|
| Training   | 12,025  | Model learning        |
| Validation | 3,463   | Hyperparameter tuning |
| Testing    | 133     | Final evaluation      |

Compatible datasets: TNO, FLIR, RoadScene, or any paired IR-VIS dataset.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ir-visible-fusion.git
cd ir-visible-fusion

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
torch>=1.10
torchvision
opencv-python
numpy
matplotlib
ultralytics

---

## Usage

**Step 1 — Preprocess images**
```bash
python preprocessing/preprocess.py --ir data/infrared --vis data/visible --out data/processed
```

**Step 2 — Fuse image pairs**
```bash
python fusion/fuse.py --input data/processed --out data/fused
```

**Step 3 — Run detection**
```bash
python detection/detect.py --input data/fused --weights yolov8n.pt --out results/qualitative
```

---

## Results

| Method                        | Precision | Recall | mAP  |
|-------------------------------|-----------|--------|------|
| Visible-only Detection        | 0.75      | 0.68   | 0.72 |
| Infrared-only Detection       | 0.72      | 0.65   | 0.70 |
| Conventional Fusion           | 0.85      | 0.78   | 0.82 |
| **Proposed Fusion + YOLO**    | **0.94**  | **0.86**| **0.92** |

The proposed method achieves a **12.2% improvement in mAP** over
conventional fusion and a **27.8% improvement** over visible-only detection.

---
