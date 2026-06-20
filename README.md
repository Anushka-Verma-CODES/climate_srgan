# SRGAN for Climate Change Monitoring
**Satellite Image Super-Resolution with Spectral Normalization and Downstream Task Evaluation**

**Author:** Anushka Verma  

---

## Overview
This project implements a modified SRGAN (Super-Resolution GAN) for enhancing 
low-resolution satellite imagery from the EuroSAT dataset. Key contributions:
1. Spectral normalization in discriminator for satellite texture stability
2. Downstream ResNet18 classification as primary evaluation metric
3. Grad-CAM explainability on discriminator

---

## Dataset
EuroSAT (Sentinel-2 satellite imagery)  
Download: https://madm.dfki.de/files/sentinel/EuroSAT.zip  
Classes used: Forest, SeaLake, HerbaceousVegetation, River

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

## Run
Open `climate_srgan.ipynb` in VS Code or Jupyter and run cells in order.

---

## Results
| Method  | PSNR (dB) | SSIM   | Classification Confidence |
|---------|-----------|--------|--------------------------|
| Bicubic | 54.28     | 0.9964 | 0.3556                   |
| SRGAN   | 46.43     | 0.9886 | 0.3605 (+1.38%)          |

Note: Bicubic PSNR is artificially high due to circular evaluation.
SRGAN trades PSNR for perceptual quality (Ledig et al. 2017).

---

## References
- Ledig et al. 2017 — SRGAN
- Helber et al. 2019 — EuroSAT
- Miyato et al. 2018 — Spectral Normalization
- Selvaraju et al. 2017 — Grad-CAM
