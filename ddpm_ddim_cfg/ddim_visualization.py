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


def show_prediction_fid(valloader, scheduler, model, device, eta,  out_dir="checkpoints/val_samples", cfg=True, cfg_weight=2.5):
    
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, "real")
    gen_dir  = os.path.join(out_dir, "gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)
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
            for t in range(scheduler.inference_step-1,-1,-1): ## 99, 98 ...
                t_tensor = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
                if cfg:
                    cond_noise = model(x_t, t_tensor, cls)
                    uncond_noise = model(x_t, t_tensor, torch.zeros_like(cls))
                    noise = (1 + cfg_weight) * cond_noise - cfg_weight * uncond_noise
                else:
                    noise = model(x_t, t_tensor)

                _, x_t_1, __ = scheduler.reverse_process(t=t_tensor, x_t=x_t, eps=noise, eta=eta)
                x_t = x_t_1

            gen_imgs = (x_t + 1) / 2 
            for i in range(gen_imgs.size(0)):
                save_image(gen_imgs[i].cpu(), os.path.join(gen_dir, f"{save_idx:06d}.png"))

            save_idx += img.size(0)

    print(f"저장 완료: {real_dir}, {gen_dir}")
    return real_dir, gen_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda"
    
    valset = data.dataloader.CustomDataset(test=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=16,num_workers=4,shuffle=False)

    model = DiffusionUnet(cfg=False).to(device)
    ddim_scheduler = DDIMScheduler(inference_step=1000,device=device)
    ddim_scheduler.set_time(inference_step=100)
    
    #print(ddim_scheduler.cumprod_alpha[-5:])
    #print(ddim_scheduler.alpha[-5:])
    
    model.load_state_dict(torch.load("checkpoints/DDPM.pth", map_location=device))

    model.eval()
    

    show_prediction_fid(
        valloader,
        ddim_scheduler,
        model,
        device,
        eta=args.eta,
    )
 
