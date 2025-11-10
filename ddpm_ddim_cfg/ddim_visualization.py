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



def show_prediction(valloader, scheduler, model, device, eta,  out_dir="checkpoints/val_samples/examples", cfg=False, cfg_weight=2.5,inference_step=100):
    img, cls = next(iter(valloader))
    img = img.to(device)
    cls = cls.to(device)
    img = img * 2 - 1
    t_len = scheduler.inference_step
    x_t = torch.randn_like(img)
    
    snap_idxs = torch.linspace(0, t_len - 1, steps=1).round().long().tolist()
    snap_idxs = set(int(i) for i in snap_idxs)
    snapshots = [] 
    
    model.eval()
    with torch.no_grad():
        for t in range(scheduler.inference_step-1,-1,-1):
            t= torch.full((img.shape[0],), t, device=device, dtype=torch.long) ## 이러면 t는 배치사이즈
            
            if cfg :
                cond_noise = model(x_t,t,cls)
                uncond_noise = model(x_t,t,torch.zeros_like(cls))
                noise = (1+cfg_weight)*cond_noise - cfg_weight * uncond_noise
            
            else :
                noise = model(x_t,t)
            _,x_t_1,__ = scheduler.reverse_process(t=t,x_t=x_t,eps=noise, eta=eta)
            x_t = x_t_1
        
            t_idx_int = int(t[0].item())
            if t_idx_int in snap_idxs:
                x_t_1 = (x_t_1 +1 )/2
                snapshots.append(x_t_1[:min(8, x_t_1.size(0))])

    samples = torch.cat(snapshots, dim=-1)
    grid = make_grid(samples, nrow=1, normalize=False)
    os.makedirs(out_dir, exist_ok=True)

    img_path = os.path.join(out_dir, f"step_{inference_step}_eta_{eta}.png")
    save_image(grid, img_path)
                            
    return x_t, img_path


def show_prediction_fid(valloader, scheduler, model, device, eta,  out_dir="checkpoints/val_samples", cfg=False, cfg_weight=2.5,inference_step=100):
    
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, f"step_{inference_step}_eta_{eta}_real")
    gen_dir  = os.path.join(out_dir, f"step_{inference_step}_eta_{eta}_gen")
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
            for t in tqdm(range(scheduler.inference_step-1,-1,-1)): ## 99, 98 ...
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

def show_prediction_DDPM_fid(valloader, scheduler, model, device, eta,  out_dir="checkpoints/val_samples", cfg=False, cfg_weight=2.5,inference_step=100):
    
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, f"real")
    gen_dir  = os.path.join(out_dir, f"gen_{cfg_weight}")
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

                _, x_t_1, __ = scheduler.reverse_process(t=t_tensor, x_t=x_t, eps=noise)
                x_t = x_t_1

            gen_imgs = (x_t + 1) / 2 
            for i in range(gen_imgs.size(0)):
                save_image(gen_imgs[i].cpu(), os.path.join(gen_dir, f"{save_idx:06d}_cfgweight_{cfg_weight}.png"))

            save_idx += img.size(0)

    print(f"저장 완료: {real_dir}, {gen_dir}")
    return real_dir, gen_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--inference_step",type=int, default=100)
    parser.add_argument("--model",type=str, default="ddpm")
    parser.add_argument("--cfg", action="store_true")
    parser.add_argument("--cfg_weight", type=float, default= 2.5)
    args = parser.parse_args()

    device = "cuda"
    
    valset = data.dataloader.CustomDataset(test=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=16,num_workers=4,shuffle=False)


    
    ## for ddpm cfg
    if args.model == "ddpm" :
        model = DiffusionUnet(cfg=True).to(device)
        scheduler = DDPMScheduler(inference_step=1000,device=device)
        model.load_state_dict(torch.load("checkpoints/DDPM_CFG.pth", map_location=device))
        model.eval()
        #show_prediction(valloader,scheduler,model,device,eta=args.eta,inference_step=args.inference_step,cfg=args.cfg, cfg_weight=args.cfg_weight)
        show_prediction_DDPM_fid(valloader,scheduler,model,device,eta=args.eta,inference_step=args.inference_step,cfg=args.cfg, cfg_weight=args.cfg_weight)
               
    else :
        model = DiffusionUnet(cfg=False).to(device)
        scheduler = DDIMScheduler(inference_step=1000,device=device)
        scheduler.set_time(inference_step=args.inference_step)
        model.load_state_dict(torch.load("checkpoints/DDPM.pth", map_location=device))
        model.eval()
        show_prediction(valloader,scheduler,model,device,eta=args.eta,inference_step=args.inference_step)
        show_prediction_fid(valloader,scheduler,model,device,eta=args.eta,inference_step=args.inference_step)
        
    #print(ddim_scheduler.cumprod_alpha[-5:])
    #print(ddim_scheduler.alpha[-5:])
    
   

    

 
