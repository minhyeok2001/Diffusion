import data.get_data 
import data.dataloader 
from .diffusion_model import *
from .loss import *
from .utils import *
import argparse

import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torchvision.utils import make_grid, save_image


def show_prediction_mu_fid(valloader, ddpm_scheduler, model, device, out_dir="checkpoints/mu/val_samples", cfg=True, cfg_weight=2.5):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, "real")
    gen_dir  = os.path.join(out_dir, "gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)

    t_len = len(ddpm_scheduler.timesteps)
    save_idx = 0

    with torch.no_grad():
        for img, cls in tqdm(valloader):
            img = img.to(device)
            cls = cls.to(device)

            img_scaled = img * 2 - 1

            real_imgs = img.detach().cpu()
            for i in range(real_imgs.size(0)):
                save_image(real_imgs[i], os.path.join(real_dir, f"{save_idx:06d}.png"))

            x_t = torch.randn_like(img_scaled) 
            for t in range(t_len - 1, -1, -1):
                t = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
                
                mu = model(x=x_t,t=t,cls=cls)
                _,x_t_1,__ = ddpm_scheduler.reverse_process_mu(t=t,x_t=x_t,mu=mu)
                x_t = x_t_1

            gen_imgs = (x_t + 1) / 2 
            for i in range(gen_imgs.size(0)):
                save_image(gen_imgs[i].cpu(), os.path.join(gen_dir, f"{save_idx:06d}.png"))

            save_idx += img.size(0)

    print(f"저장 완료: {real_dir}, {gen_dir}")
    return real_dir, gen_dir

def show_prediction_x_0_fid(valloader, ddpm_scheduler, model, device, out_dir="checkpoints/x_0/val_samples", cfg=True, cfg_weight=2.5):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, "real")
    gen_dir  = os.path.join(out_dir, "gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)

    t_len = len(ddpm_scheduler.timesteps)
    save_idx = 0

    with torch.no_grad():
        for img, cls in tqdm(valloader):
            img = img.to(device)
            cls = cls.to(device)

            img_scaled = img * 2 - 1

            real_imgs = img.detach().cpu()
            for i in range(real_imgs.size(0)):
                save_image(real_imgs[i], os.path.join(real_dir, f"{save_idx:06d}.png"))

            x_t = torch.randn_like(img_scaled) 
            for t in range(t_len - 1, -1, -1):
                t= torch.full((img.shape[0],), t, device=device, dtype=torch.long)

                x_0 = model(x=x_t,t=t,cls=cls)
                _,x_t_1,__ = ddpm_scheduler.reverse_process_x_0(t=t,x_t=x_t,x_0=x_0)
                x_t = x_t_1

            gen_imgs = (x_t + 1) / 2 
            for i in range(gen_imgs.size(0)):
                save_image(gen_imgs[i].cpu(), os.path.join(gen_dir, f"{save_idx:06d}.png"))

            save_idx += img.size(0)

    print(f"저장 완료: {real_dir}, {gen_dir}")
    return real_dir, gen_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_step",type=int, default=100)
    args = parser.parse_args()

    device = "cuda"
    
    valset = data.dataloader.CustomDataset(test=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=16,num_workers=4,shuffle=False)

    model = DiffusionUnet(cfg=True).to(device)
    scheduler = DDPMScheduler(inference_step=1000,device=device)
    model.load_state_dict(torch.load("checkpoints/Diffusion_mu.ckpt", map_location=device))
    model.eval()
    show_prediction_mu_fid(valloader,scheduler,model,device)
    
    model = DiffusionUnet(cfg=True).to(device) 
    scheduler = DDPMScheduler(inference_step=1000,device=device)
    model.load_state_dict(torch.load("checkpoints/Diffusion_x_0.pth", map_location=device))
    model.eval()
    show_prediction_x_0_fid(valloader,scheduler,model,device)