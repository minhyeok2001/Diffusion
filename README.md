# 🌫️ Project overview

This project focuses on generative modeling, with the main goal of implementing diffusion-related models from scratch.
It covers a variety of models — *VAE, DDPM/DDIM, CFG, Stable Diffusion, DPM-Solver, and Flow Matching* — all trained on a single dataset, AFHQ.
Each model will be evaluated and compared using the FID score to assess generation quality.

**All implementations are written entirely from scratch, without using any pretrained models or code generated/copied from GPT.**



# Dataset
AFHQ (Animal Faces-HQ) consists of 16,130 high-quality images at 512×512 resolution. Since this is a generative modeling project (not classification), minimal preprocessing is applied.
<img width="2354" height="337" alt="image" src="https://github.com/user-attachments/assets/287be022-c4ba-4157-b4cd-24d0de5691ca" />


# Directory

```bash
Diffusion-From-Scratch/
│
├── data/
│   ├── afhq/                # AFHQ dataset
│   └── dataloader.py        
```

  

# Contents
- [1. VAE](#Variantional Auto Encoder)
- [2. DDPM, DDIM, CFG](#ddpmddimcfg)
- [3. Stable Diffusion](#stable-diffusion)
- [4. DPM-Solver](#dpm-solver)


# Experimental Results

