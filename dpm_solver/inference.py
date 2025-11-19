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

"""
flow 
1. ODE sovler class를 만들어서, 여기서 reverse process를 정의하기. ( 1차, 2차 , 3차 ... ) -> args로 받기
2. 이 외, 모델은 DDPM_cfg pth를 활용
3. sampling 후에 FiD score 비교
"""

def run(args):
    device = "mps"
    num_order = args.num_order
    inference_step = args.inference_step
    cfg_weight = args.cfg_weight
    
    model = DiffusionUnet(cfg=True).to(device)
    model.load_state_dict(torch.load("checkpoints/DDPM_CFG.pth",map_location=device),strict=True)
    
    dpm_solver = DpmSolver(device=device,inference_step=inference_step)
    
    valset = data.dataloader.CustomDataset(test=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=4,num_workers=4,shuffle=False)
    
    os.makedirs("checkpoints/samples", exist_ok=True)
    model.eval()
    with torch.no_grad():
        for img, cls in tqdm(valloader):
            img = img.to(device)
            cls = cls.to(device)

            img = img * 2 - 1

            x_s = torch.randn_like(img)
            for i in torch.arange(inference_step-1, 0, -1):
                s = i * dpm_solver.ratio
                t = torch.clamp((i-1) * dpm_solver.ratio,0)

                s = torch.full((img.shape[0],),s,device=device, dtype=torch.long) 
                t = torch.full((img.shape[0],),t,device=device, dtype=torch.long) 
                
                ## CFG network
                cond_noise = model(x_s,s,cls)
                uncond_noise = model(x_s,s,torch.zeros_like(cls))
                noise = (1+cfg_weight)*cond_noise - cfg_weight * uncond_noise
                
                if num_order == 1:
                    x_t = dpm_solver.first_order(t=t,s=s,x_s=x_s,eps=noise)
                
                else :
                    raise RuntimeError("NOT IMPLEMENTED YET !!")
                
                x_s = x_t
            
            x_vis = (x_s + 1) / 2
            x_vis = torch.clamp(x_vis, 0, 1)

            grid = make_grid(x_vis, nrow=8)
            save_image(grid, f"checkpoints/samples/sample_{inference_step}.png")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_order", type=int)
    parser.add_argument("--inference_step", type=int)
    parser.add_argument("--cfg_weight", type=float,default=2.5)
    args = parser.parse_args()
    run(args)


