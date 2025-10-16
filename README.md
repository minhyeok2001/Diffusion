# 🌫️ Project overview

This project focuses on generative modeling, with the main goal of implementing diffusion-related models from scratch.

It covers a variety of models such as 

— *VAE, DDPM/DDIM, CFG, Stable Diffusion, DPM-Solver, and Flow Matching* — 

which are all trained on a single dataset, AFHQ.

Each model will be evaluated and compared using the FID score to assess generation quality. 

All the equations required for each model are derived and explained in the README files within their respective directories.



**🔥All implementations are written entirely from scratch, without using any pretrained models or code generated/copied from GPT🔥**



# 🐾 Dataset
AFHQ (Animal Faces-HQ) consists of 16,130 high-quality images at 512×512 resolution. Since this is a generative modeling project (not classification), minimal preprocessing is applied.
<img width="2354" height="337" alt="image" src="https://github.com/user-attachments/assets/287be022-c4ba-4157-b4cd-24d0de5691ca" />

cat : 5153, dog : 4739, wild : 4738

# 📁 Directory

```bash
Diffusion/
│
├── data/
│   ├── dataset
│   └── dataloader.py        
```

# Training Setup

**Environment** : 

Use A100 on google colab, ~

# 📚 Contents
- [1. VAE](vae/)
- [2. DDPM, DDIM, CFG](ddpm_ddim_cfg/)
- [3. Stable Diffusion](stable_diffusion/)
- [4. DPM-Solver](dpm_solver/)


# 📈 Experimental Results



