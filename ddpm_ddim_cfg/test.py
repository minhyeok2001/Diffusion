import data.get_data 
import data.dataloader 
from .diffusion_model import *
from .loss import *
from .utils import *

import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

import diffusers
from diffusers import UNet2DModel

batch_size = 1
device = "mps"
cfg = False
hola = 1000

dataset = data.dataloader.CustomDataset()
trainloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,collate_fn=data.dataloader.collate_ft,shuffle=True)

model = DiffusionUnet(cfg=cfg).to(device)
ddpm_scheduler = DDPMScheduler(inference_step=hola,device=device)

print("my params :",sum(p.numel() for p in model.parameters()))

hola = torch.randn(6,3,128,128).to(device)
t = torch.randint(0,1000,(6,)).to(device)

print(model(hola,t).shape)

"""
official_model = UNet2DModel.from_pretrained("faverogian/Smithsonian128UNet").to(device)
official_ddpm_scheduler = diffusers.DDPMScheduler(num_train_timesteps=hola,beta_start=1e-4,beta_end=2e-2)

print("official params :",sum(p.numel() for p in official_model.parameters()))

img,cls = next(iter(trainloader)) ## 3x3x128x128

img = img.to(device)

def forward_image(img,official=False):
    img_stack = []
    for i in range(0,hola,100):
        t = torch.full((img.shape[0],), i,device=device)
        eps =  torch.randn_like(img)
        if official :
            forward_img = official_ddpm_scheduler.add_noise(img,eps,t)
        else :
            forward_img , noise = ddpm_scheduler.forward_process(t,img,eps)
        img_stack.append(forward_img)
    return img_stack

def reverse_image(img):
    test_reverse = TestReverse()
    img_stack = []
    idx = [i for i in range(0,hola,100)]
    for i in tqdm(range(hola-1,-1,-1)):
        t = torch.full((img.shape[0],),i,device=device)

        _,reverse_image,_ = test_reverse.reverse_process(t,img,official_model(img,t).sample)
        if i in idx :
            img_stack.append(reverse_image)
        
    return img_stack


class TestReverse(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        timesteps = torch.arange(1000-1,-1,-1,device=device) ## Diffuser에서 확인한 결과, 1000이면 999~0임
        beta = torch.linspace(1e-4,2e-2,1000,device=device) ## 공식 논문 베타 값 기준
        alpha = 1-beta
        cumprod_alpha = torch.cumprod(alpha,-1)
        
        #print("cum",cumprod_alpha)
        
        ## register buffer 활용하여, 이후 체크포인트에서도 사용 가능하게
        self.register_buffer("timesteps",timesteps)
        self.register_buffer("alpha",alpha)
        self.register_buffer("cumprod_alpha",cumprod_alpha)
    
    def teeth(self,const,t):
        const = const.to(t.device)
        return const.gather(-1,t).reshape(-1,1,1,1)

    def reverse_process(self,t,x_t,eps,noise=None):
        
        self.cumprod_alpha= official_ddpm_scheduler.alphas_cumprod.to(x_t.device)
        self.alpha= official_ddpm_scheduler.alphas.to(x_t.device)
        
        alpha_bar = self.teeth(self.cumprod_alpha,t)
        alpha = self.teeth(self.alpha,t)
        
        
        mu = (1 / torch.sqrt(alpha)) * (x_t - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps)
        alpha_bar_prev = self.teeth(torch.cat([torch.tensor([1.0],device=x_t.device),self.cumprod_alpha[:-1]],dim=0),t)
        
        sigma_square = ((1-alpha_bar_prev) / (1-alpha_bar)) * (1-alpha)

        if noise is None:
            noise = torch.randn_like(x_t)
            
        sample_prev = mu + torch.sqrt(sigma_square) * noise

        return mu, sample_prev, noise

    def check_same(self,):
        self.official_cumprod_alpha= official_ddpm_scheduler.alphas_cumprod.to(device)
        self.official_alpha= official_ddpm_scheduler.alphas.to(device)
        
        print("cum_alpha 같음? : ",torch.allclose(self.official_cumprod_alpha,self.cumprod_alpha))
        print("alpha 같음? : ",torch.allclose(self.official_alpha,self.alpha))
        
        print("===============alpha 10까지===============")
        print(self.alpha[:10])
        print(self.alpha[-10:])
        print("===============official alpha 10까지===============")
        print(self.official_alpha[:10])
        print(self.official_alpha[-10:])
        

def show_tensor_images(x):
    if isinstance(x, list):
        x = torch.cat(x, dim=0)

    if x.ndim == 3:
        x = x.unsqueeze(0)

    x = x.detach().cpu()
    if x.min() < 0:
        x = (x + 1) / 2

    x = x.clamp(0, 1)

    chunks = [x[i:i+3] for i in range(0, x.size(0), 3)]
    col_images = [make_grid(chunk, nrow=1) for chunk in chunks]
    grid = torch.cat(col_images, dim=2)  
    
    
    # 파일 저장
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "reverse_result.png")
    save_image(grid, save_path)

    print(f"✅ Saved reverse diffusion result to: {save_path}")
    return save_path


## 확인결과 forward는 문제 x

#show_tensor_images(img)
#show_tensor_images(forward_image(img))
#show_tensor_images(forward_image(img,official=True))

## reverse 식도 문제 x 


## scheduler도 문제 x

#show_tensor_images(reverse_image(img))
#test_reverse = TestReverse()
#test_reverse.check_same()

"""