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

from tqdm import tqdm
from torchvision.utils import make_grid, save_image


def show_prediction(step,valloader,ddpm_scheduler,model,device,out_dir="checkpoints/val_samples"):
    img, cls =next(iter(valloader))
    img = img.to(device)
    cls = cls.to(device)
    img = img * 2 - 1
    t_len = len(ddpm_scheduler.timesteps)
    x_t = torch.randn_like(img) ## 어차피 처음엔 노이즈니까 이렇게 고고 
    
    snap_idxs = torch.linspace(0, t_len - 1, steps=10).round().long().tolist()
    snap_idxs = set(int(i) for i in snap_idxs)
    snapshots = [] 
    
    model.eval()
    with torch.no_grad():
        for t in range(t_len-1,-1,-1):
            t= torch.full((img.shape[0],), t, device=device, dtype=torch.long) ## 이러면 t는 배치사이즈
            noise = model(x_t,t)
            _,x_t_1,__ = ddpm_scheduler.reverse_process(t=t,x_t=x_t,eps=noise)
            x_t = x_t_1
        
            t_idx_int = int(t[0].item())
            if t_idx_int in snap_idxs:
                x_t_1 = (x_t_1 +1 )/2
                snapshots.append(x_t_1[:min(8, x_t_1.size(0))])

    samples = torch.cat(snapshots, dim=-1)
    grid = make_grid(samples, nrow=1, normalize=False)
    os.makedirs(out_dir, exist_ok=True)
    save_image(grid, os.path.join(out_dir, f"iter_{step}_timeline.png"))
                            
    return x_t

def run(args):
    device = "cuda"
    epoch = args.epoch 
    lr = args.lr 
    batch_size = args.batch_size
    num_workers = args.num_workers
    cfg = args.cfg
    
    dataset = data.dataloader.CustomDataset()
    trainloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,collate_fn=data.dataloader.collate_ft,num_workers= num_workers)
    
    valset = data.dataloader.CustomDataset(test=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=batch_size,num_workers= num_workers,shuffle=False)
    
    ## 2. Model definition & setting stuffs..
    
    model = DiffusionUnet(cfg=cfg).to(device)
    ddpm_scheduler = DDPMScheduler(inference_step=1000,device=device)

    print("model params : ",sum(item.numel() for item in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=epoch)
    
    loss_ft = DiffusionLoss()
    
    checkpoint_path = "checkpoints/Diffusion.pth"
    
    sample_dir = "checkpoints/val_samples"
    os.makedirs(sample_dir, exist_ok=True)

    ## 3. train loop
    ## method : 배치사이즈만큼의 time step을 랜덤으로 만든다 -> 해당 타임스텝에서의 forward process를 가져온다 -> 그걸 넣고 노이즈를 예측하도록 한다 
    
    for i in range(epoch) :
        model.train()
        running_loss = 0.0
        total_len = len(trainloader)
        for img, cls in tqdm(trainloader) :
            optimizer.zero_grad()
            
            img = img.to(device)
            cls = cls.to(device)
            
            ## diffusion도 vae와 마찬가지로 입력을 -1~1 바꾸기. 어차피 마지막 layer에 tanh도 없으므로 ..
            img = img * 2 - 1
            
            ## 1. timestep을 만든다
            ### 아하 !! 우리는 그 collate_fn 직접 만들어서 3개 동시에 넣어줬으니까, 이거 배치사이즈로 만들면 안되고 3 곱해서 해야지. 실제로 배치사이즈가 3이면 9개 이미지 들어가는거니까
            t_idx =torch.randint(0,len(ddpm_scheduler.timesteps),(img.shape[0]*3,), device=device)

            ## 2. 해당 t에 맞게 forward process를 한다 with noise_gt
            x_t, noise_gt = ddpm_scheduler.forward_process(t=ddpm_scheduler.timesteps[t_idx],x_0=img)

            ## 3. noise 예측 Unet
            noise_pred = model(x=x_t,t=ddpm_scheduler.timesteps[t_idx])

            loss = loss_ft(noise_pred,noise_gt)
            
            loss.backward()
            optimizer.step()
            
            print("loss : ", loss.item())
            running_loss += loss.item()
 
        avg_train_loss = running_loss / total_len
        print(f"Epoch [{i+1}/{epoch}] | Train Loss: {avg_train_loss:.6f}")
    
        
        val_loss = 0.0
        val_batches = len(valloader)
        with torch.no_grad():
            for idx,(img, cls) in tqdm(enumerate(valloader)):
                model.eval()
                img = img.to(device)
                cls = cls.to(device)
                
                img = img * 2 - 1
                
                t_idx =torch.randint(0,len(ddpm_scheduler.timesteps),batch_size, device=device)
                
                x_t, noise_gt = ddpm_scheduler.forward_process(t=ddpm_scheduler.timesteps[t_idx],x_0=img)
                
                noise_pred = model(x=x_t,t=ddpm_scheduler.timesteps[t_idx])

                loss = loss_ft(noise_pred,noise_gt)

                val_loss += loss.item()
            
        avg_val_loss = val_loss / val_batches
        print(f"Epoch [{i+1}/{epoch}] | Val Loss: {avg_val_loss:.6f}")
        scheduler.step()
        
        show_prediction(step=i,valloader=valloader,ddpm_scheduler=ddpm_scheduler,model=model,device=device)


    torch.save(model.state_dict(), checkpoint_path)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--cfg", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1) 
    
    args = parser.parse_args()
    
    run(args)
