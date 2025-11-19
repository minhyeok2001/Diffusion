import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import wandb

import data.get_data 
import data.dataloader 
from .utils import DpmSolver
from ddpm_ddim_cfg.diffusion_model import DiffusionUnet
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

def run(args):
    device = "cuda"
    num_order = args.num_order
    inference_step = args.inference_step
    cfg_weight = args.cfg_weight
    
    model = DiffusionUnet(cfg=True).to(device)
    model.load_state_dict(torch.load("checkpoints/DDPM_CFG.pth",map_location=device),strict=True)
    
    dpm_solver = DpmSolver(device=device,inference_step=inference_step)
    
    valset = data.dataloader.CustomDataset(test=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=4,num_workers=4,shuffle=False)
    
    out_dir =f"checkpoints/first_step_{inference_step}"
    
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, "real")
    gen_dir  = os.path.join(out_dir, "gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)
    
    model.eval()

    save_idx = 0
    with torch.no_grad():
        for img, cls in tqdm(valloader):
            img = img.to(device)
            cls = cls.to(device)

            img = img * 2 - 1
            
            real_imgs = img.detach().cpu()
            for i in range(real_imgs.size(0)):
                save_image(real_imgs[i], os.path.join(real_dir, f"{save_idx:06d}.png"))

            x_s = torch.randn_like(img)
            
            time_list = torch.linspace(dpm_solver.num_timestep-1,0,steps=inference_step,device=device,dtype=torch.long)
            
            for idx in range(len(time_list)-1):
                
                ## 이부분 수정요망 -> ratio 곱하지말고 그냥 torch.linspace 활용
                s = time_list[idx]
                t = time_list[idx+1]

                s = torch.full((img.shape[0],),s,device=device, dtype=torch.long) 
                t = torch.full((img.shape[0],),t,device=device, dtype=torch.long) 
                
                ## CFG network
                cond_noise = model(x_s,s,cls)
                uncond_noise = model(x_s,s,torch.zeros_like(cls))
                noise = (1+cfg_weight)*cond_noise - cfg_weight * uncond_noise
                
                if num_order == 1:
                    x_t = dpm_solver.first_order(t=t,s=s,x_s=x_s,eps=noise)
                
                elif num_order == 2:
                    x_t = dpm_solver.second_order(t=t,s=s,x_s=x_s,eps=noise)
                    
                elif num_order == 3:
                    x_t = dpm_solver.third_order(t=t,s=s,x_s=x_s,eps=noise)
                else :
                    raise RuntimeError("NOT IMPLEMENTED YET !!")
                
                x_s = x_t
            
            
            gen_imgs = (x_t + 1) / 2 
            for i in range(gen_imgs.size(0)):
                save_image(gen_imgs[i].cpu(), os.path.join(gen_dir, f"{save_idx:06d}.png"))
                
            save_idx += img.size(0)
            
    print(f"저장 완료: {real_dir}, {gen_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_order", type=int)
    parser.add_argument("--inference_step", type=int)
    parser.add_argument("--cfg_weight", type=float,default=2.5)
    args = parser.parse_args()
    run(args)


