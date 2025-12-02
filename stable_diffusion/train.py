import data.get_data 
import data.dataloader 
from ddpm_ddim_cfg.loss import *
from ddpm_ddim_cfg.utils import DDPMScheduler
from .sd_utils import *
from .stable_diffusion_model import StableDiffusionUnet


import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import wandb

from tqdm import tqdm
from torchvision.utils import make_grid, save_image


## 지금 cat : 1 , dog : 2 , wild : 3

def make_sentence(cls):
    if cls == 1:
        animal = "cat"
    elif cls == 2:
        animal = "dog"
    elif cls == 3:
        animal = "wild"
    else:
        raise RuntimeError("그런 동물 없습니다~")
    
    sentence = f"A photo of {animal}"
    
    return sentence



def run(args):
    wandb.login(key="08198b7be027ddffa5241b9acf2f45cd4d42e993")
    
    device = "cuda"
    epoch = args.epoch 
    lr = args.lr 
    batch_size = args.batch_size
    num_workers = args.num_workers
    cfg_dropout= args.cfg_dropout
    cfg_weight = args.cfg_weight
    
    wandb.init(
        project="Diffusion",
        config={
            "epochs": epoch,
            "lr": lr,
            "batch_size": batch_size,
            "num_workers": num_workers
        }
    )
    
    dataset = data.dataloader.CustomDataset()
    trainloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,collate_fn=data.dataloader.collate_ft,num_workers= num_workers,shuffle=True)
    
    valset = data.dataloader.CustomDataset(test=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=batch_size,num_workers=num_workers,shuffle=False)
    
    visual_valloader = torch.utils.data.DataLoader(valset,batch_size=1,num_workers=num_workers,shuffle=False)
    
    model = StableDiffusionUnet(cfg=True,cfg_dropout=cfg_dropout).to(device)
    ddpm_scheduler = DDPMScheduler(inference_step=1000,device=device)
    
    wandb.watch(model, log="all")

    print("model params : ",sum(item.numel() for item in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=epoch)
    
    loss_ft = DiffusionLoss()
    
    checkpoint_path = "checkpoints/stable_diffusion.ckpt"
    
    sample_dir = "checkpoints/val_samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    for i in range(epoch) :
        model.train()
        running_loss = 0.0
        total_len = len(trainloader)
        for img, cls in tqdm(trainloader) :
            optimizer.zero_grad()
            
            img = img.to(device)
            cls = cls.to(device)
            img = img * 2 - 1
            
            ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@비상@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            ## 지금 필요한거는, 만약 cls를 받았다면 이거를 아 이거 벡터네???? 음..... 그러니까 지금 문제가 뭐냐면, 이거 wrapper 사용해서 crossattention은 어떻게든
            ## 주입을 했는데, CLIP을 쓰면 받아오는게 vector잖아. 그러면 벡터를 먹여야하는데. -> 아하 그러면 벡터꼴이면. 즉 n.dim()이거 써가지고 벡터면 그냥 먹이고 아니면 

            clip_input = make_sentence(cls)

            t =torch.randint(0,len(ddpm_scheduler.timesteps),(img.shape[0],), device=device)
            
            x_t, noise_gt = ddpm_scheduler.forward_process(t=t,x_0=img)

            noise_pred = model(x=x_t,t=t,cls=cls)
            loss = loss_ft(noise_pred,noise_gt)    

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
 
        avg_train_loss = running_loss / total_len
        print(f"Epoch [{i+1}/{epoch}] | Train Loss: {avg_train_loss:.6f}")
        
        
        val_loss = 0.0
        val_batches = len(valloader)
        model.eval()
        with torch.no_grad():
            for idx,(img, cls) in tqdm(enumerate(valloader)):
                img = img.to(device)
                cls = cls.to(device)
                
                img = img * 2 - 1
                
                t =torch.randint(0,len(ddpm_scheduler.timesteps),(img.shape[0],), device=device)
                
                x_t, noise_gt = ddpm_scheduler.forward_process(t=t,x_0=img)

                val_loss += loss.item()
            
        avg_val_loss = val_loss / val_batches
        print(f"Epoch [{i+1}/{epoch}] | Val Loss: {avg_val_loss:.6f}")
        scheduler.step()

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": i + 1,
        })


    torch.save(model.state_dict(), checkpoint_path)
    
    
    wandb.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--cfg_dropout", type=float, default=0.2)
    parser.add_argument("--cfg_weight", type=float, default=2.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    run(args)
