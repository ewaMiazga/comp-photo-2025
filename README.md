# Emulating the Effect of Highlight Diffusion Filter

Project as part of the EPFL Computational Photography course (CS-413).

## Overview

This repository contains code and notebooks for emulating highlight diffusion filters on paired RAW images, using both physics-based kernels and deep learning models.

## Prerequisites

- Python ≥ 3.8
- [PyTorch](https://pytorch.org/) ≥ 1.12
- OpenCV
- NumPy, SciPy, Matplotlib
- [gdown](https://github.com/wkentaro/gdown) (for data download)

Install all Python dependencies:
```bash
pip install -r requirements.txt
```

## Data 

### 1. Data download

The dataset includes paired images captured with and without a highlight diffusion filter. To download and unzip:
Install all Python dependencies:

```bash
gdown 1NVK4-lKwGNDnO0CgaSWnlWoPrugl5vFA -O data/dataset_raw.zip
unzip data/dataset_raw.zip -d data/raw
rm data/dataset_raw.zip
```
### 2. Data alignment

Aligned image pairs are required for accurate kernel learning and network training. Run the alignment step on the raw data:

```bash
python run_alignment.py \
  --original data/dataset_raw/long_exp \
  --filtered data/dataset_raw/filtered_long_exp \
  --short_exp data/dataset_raw/short_exp \              #optional
```
















- `run_alignment.py`: Entry point for RANSAC-based feature matching and homography estimation.
- `alignment.py`: Contains adjustable parameters for feature detection, matching thresholds, and warping.



