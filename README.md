# SRGAN for Climate Change Monitoring
**Satellite Image Super-Resolution with Spectral Normalization and Downstream Task Evaluation**

---

## Overview

This project implements a modified SRGAN (Super-Resolution Generative Adversarial Network) 
for enhancing low-resolution climate satellite imagery from the EuroSAT dataset. 

Three specific contributions are made:
1. Spectral normalization integrated into the discriminator, specifically motivated 
   for satellite imagery where homogeneous textures destabilize batch normalization
2. Downstream ResNet18 classification confidence proposed as the primary evaluation 
   metric — arguing PSNR is insufficient and misleading under synthetic pair construction
3. Grad-CAM applied to the discriminator to interpret learned spatial features

---

## Dataset

EuroSAT — Sentinel-2 satellite imagery benchmark (Helber et al. 2019)  
Download: https://madm.dfki.de/files/sentinel/EuroSAT.zip  

Classes selected (climate-relevant only):
- Forest — deforestation monitoring
- SeaLake — water body change detection  
- HerbaceousVegetation — vegetation health assessment
- River — freshwater system analysis

Total: 11,500 images | Train: 9,200 | Val: 2,300  
Task: 4x super-resolution (64x64 → 256x256)

---

## Requirements

- Python 3.14+
- PyTorch 2.11.0 + CUDA 12.8
- NVIDIA GPU (tested on RTX 5050 8GB)

---

## Setup

```bash
git clone https://github.com/Anushka-Verma-CODES/climate_srgan
cd climate_srgan
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy matplotlib scikit-image pillow tqdm
```

---

## Usage

Open `climate_srgan.ipynb` in VS Code or Jupyter Notebook.  
Select `.venv` as kernel. Run cells in order:


| Cell | Description |
|------|-------------|
| Cell 1 | Title, overview, objectives, research questions |
| Cell 2 | Literature review and gap analysis |
| Cell 3 | Methodology and pipeline |
| Cell 4 | Environment setup and imports |
| Cell 5 | EuroSAT dataset download |
| Cell 6 | Locate extracted dataset |
| Cell 7 | HR/LR pair preparation and visualization |
| Cell 8 | Model architecture description |
| Cell 9 | Generator, Discriminator, PerceptualLoss |
| Cell 10 | Training loop (40 epochs) |
| Cell 11 | Bicubic baseline evaluation |
| Cell 12 | Visual comparison: LR vs Bicubic vs SRGAN vs HR |
| Cell 13 | Per-class PSNR/SSIM evaluation |
| Cell 14 | Downstream ResNet18 classification confidence |
| Cell 15 | Grad-CAM discriminator explainability |
| Cell 16 | Results summary |
---

## Results

### Quantitative Evaluation

| Method           | PSNR (dB) | SSIM   |
|------------------|-----------|--------|
| Bicubic Baseline | 54.28     | 0.9964 |
| SRGAN (Ours)     | 46.43     | 0.9886 |

> **Note:** Bicubic PSNR is artificially high due to circular evaluation —  
> HR ground truth was itself created via bicubic upsampling.  
> SRGAN deliberately trades PSNR for perceptual quality (Ledig et al. 2017:  
> SRGAN MOS 3.56 vs Bicubic MOS 1.47 despite lower PSNR).

### Per-Class PSNR

| Class                | SRGAN PSNR | Bicubic PSNR |
|----------------------|-----------|-------------|
| Forest               | 46.73     | 53.55       |
| SeaLake              | 45.31     | 57.72       |
| HerbaceousVegetation | 46.69     | 49.23       |
| River                | 47.10     | 49.36       |

SeaLake shows the largest gap — homogeneous water surfaces make  
bicubic circular evaluation most extreme.

### Downstream Classification (Primary Metric)

| Method           | Mean Confidence | Improvement |
|------------------|----------------|-------------|
| Bicubic          | 0.3556         | baseline    |
| SRGAN (Ours)     | 0.3605         | +1.38%      |

SRGAN preserves more climatologically meaningful spectral features  
than bicubic interpolation as measured by ResNet18 confidence.

### Training

- 40 epochs | Batch size 16 | RTX 5050 8GB
- Generator loss: 0.0275 → 0.0011 (stable convergence)
- Discriminator loss: stable ~0.689 (healthy adversarial balance)
- Spectral normalization prevented gradient collapse on homogeneous textures

---

## Project Structure

```
climate_srgan/
├── climate_srgan.ipynb     # Main notebook
├── data/
│   ├── raw/                # EuroSAT download
│   ├── train/HR            # 9,200 HR training images
│   ├── train/LR            # 9,200 LR training images
│   ├── val/HR              # 2,300 HR validation images
│   └── val/LR              # 2,300 LR validation images
├── outputs/
│   ├── checkpoints/        # Model checkpoints (every 10 epochs)
│   ├── results/            # Training curves, visual comparisons
│   └── gradcam/            # Discriminator attention maps
├── README.md
└── .gitignore
```

---

## References

- Ledig et al. (2017). Photo-Realistic Single Image Super-Resolution Using a GAN. CVPR.
- Helber et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark. IEEE JSTARS.
- Miyato et al. (2018). Spectral Normalization for GANs. ICLR.
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
- Dong et al. (2016). Image Super-Resolution Using Deep CNNs. IEEE TPAMI.
