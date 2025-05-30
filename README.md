# Emulating the Effect of Highlight Diffusion Filter

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

## Physical based approach

In the `physical_based` folder, you will find our different analysis and approaches to emulate the effect of a diffusion filter with traditional methods.

Namely, we explore the per-pixel brightness variance, an **adaptive Gaussian blur** with a pluggable function for sigma, and more importantly, we find a quite effective **overexposure extention blur method**.


## Deep learning  approach
In the `deep_learning_based` folder, you will find our method to emulate the effect of a diffusion filter using deep learning.

To run the training with N epochs, do:
```bash
python deep_learning_based/train.py --epochs N
```

## Additional analysis notebooks
- `alignment_quantification.ipynb`: this notebook contains analysis of the effectiveness of our alignment pipeline.
- `dataset_visualization.ipynb`: this notebook is useful to get an overall view of the dataset


## Structure of the repo
```
├── analysis_notebooks/
│   ├── alignment_quantification.ipynb
│   └── dataset_visualization.ipynb
├── deep_learning_based/
│   ├── direct-gaussian-optimization.ipynb
│   ├── extension_blur_net.py
│   ├── gaussian_net.py
│   ├── test.ipynb
│   ├── train_extension_blur_net.py
│   └── train.py
│   
├── physical_based/
│   ├── stats-analysis/
│   ├── blur_kernel_analysis.ipynb
│   ├── gaussian_adaptive_blur.ipynb
│   ├── gaussian_kernel.ipynb
│   ├── overexposure_extention_blur.ipynb
│   ├── variance_brightness_analysis.ipynb
│   └── variance_brightness_analysis.py
├── utils/
│   ├── alignment.py
│   ├── dataset_navigation.py
│   ├── post_processor.py
│   └── raw_utils.py
├── .gitignore
├── LICENSE
├── README.md
└── run_alignment.py
```

## Authors
- Nour Guermazi ([@nourguermazi01](https://github.com/nourguermazi01))  
- Ewa Miazga ([@ewaMiazga](https://github.com/ewaMiazga))  
- Gunnar Dofri Vidarsson ([@GDofri](https://github.com/GDofri))  
- Boris Zhestiankin ([@zhestyatsky](https://github.com/zhestyatsky))  

Supervised by: [Liying Lu](https://people.epfl.ch/liying.lu) 
As part of CS-413: Computational Photography at EPFL, taught by [Sabine Süsstrunk](https://people.epfl.ch/sabine.susstrunk) ([IVRL](https://www.epfl.ch/labs/ivrl/))

