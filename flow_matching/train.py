import data.get_data 
import data.dataloader 
from .flow_matching_model import *
from .utils import *
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import wandb

from tqdm import tqdm
import torchvision
from torchvision.utils import make_grid, save_image

def run(args):
    wandb.login(key="08198b7be027ddffa5241b9acf2f45cd4d42e993")
    
    device = "mps"
    epoch = args.epoch 
    lr = args.lr 
    batch_size = args.batch_size
    num_workers = args.num_workers
    sigma_min = args.sigma_min
    inference_step = args.inference_step
    model_type = args.model_type

    
    wandb.init(
        project="FlowMatching",
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
    ## 2. Model definition & setting stuffs..
    
    model = FlowMatchingUnet().to(device)
    fm_scheduler = FlowMatchingScheduler(model_type=model_type).to(device)
    
    wandb.watch(model, log="all")

    print("model params : ",sum(item.numel() for item in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=epoch)
    
    loss_ft = FlowMatchingLoss()
    
    checkpoint_path = "checkpoints/FlowMatching.pth"
    
    sample_dir = "checkpoints/val_samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    for i in range(epoch) :
        model.train()
        running_loss = 0.0
        total_len = len(trainloader)
        for img, cls in tqdm(trainloader) :
            optimizer.zero_grad()
            img = img.to(device)
            img = img * 2 - 1
            
            ## 1. 학습할 timestep t 정하기.
            t = torch.rand((img.shape[0],),dtype=torch.float,device=device)
            
            ## 2. x_t에서의 vector field 예측하기
            x_0 = torch.randn_like(img)
            x_t = fm_scheduler.flow_map(t=t,x_1=img,sigma_min=sigma_min,x_0=x_0)
            pred = model(x_t,t)
            
            ## 3. GT vector field 구하기
            gt = fm_scheduler.vector_field(t=t,x_1=img,sigma_min=sigma_min,x_0=x_0)  ## 벡터필드 내부에서 flowmap 가져다 쓰도록 해놨음
            loss = loss_ft(pred,gt)
            
            loss.backward()
            optimizer.step()
            
            #print("loss : ", loss.item())
            running_loss += loss.item()
 
        avg_train_loss = running_loss / total_len
        print(f"Epoch [{i+1}/{epoch}] | Train Loss: {avg_train_loss:.6f}")
        
        
        val_loss = 0.0
        val_batches = len(valloader)
        model.eval()
        with torch.no_grad():
            for idx,(img, cls) in tqdm(enumerate(valloader)):
                img = img.to(device)
                img = img * 2 - 1
            
                ## 1. 학습할 timestep t 정하기.
                t = torch.rand((img.shape[0],),dtype=torch.float,device=device)
                
                ## 2. x_t에서의 vector field 예측하기
                x_0 = torch.randn_like(img)
                x_t = fm_scheduler.flow_map(t=t,x_1=img,sigma_min=sigma_min,x_0=x_0)
                pred = model(x_t,t)
                
                ## 3. GT vector field 구하기
                gt = fm_scheduler.vector_field(t=t,x_1=img,sigma_min=sigma_min,x_0=x_0)
                loss = loss_ft(pred,gt)

                val_loss += loss.item()
            
        avg_val_loss = val_loss / val_batches
        print(f"Epoch [{i+1}/{epoch}] | Val Loss: {avg_val_loss:.6f}")
        scheduler.step()
        
        ##epoch마다 실제 inference한번 돌려보기
        
        with torch.no_grad():
            
            img,cls = next(iter(visual_valloader))
            
            img = img.to(device)
            img = img*2-1
            
            timestep = torch.linspace(0,1-(1e-8),inference_step,dtype=torch.float,device=device)
            save_idx = torch.linspace(0, inference_step - 1, steps=5).long().tolist()
            
            x_t = torch.randn_like(img)
            snapshots = []
            
            for test_idx,t in enumerate(timestep) :
                if test_idx < inference_step-1 :
                    delta_t = timestep[test_idx+1]-timestep[test_idx]
                x_t = fm_scheduler.sampling(model,t,x_t,delta_t)
                
                if test_idx in save_idx:
                    x_out = (x_t + 1) / 2
                    snapshots.append(x_out)
                
        samples = torch.cat(snapshots, dim=-1)
        grid = make_grid(samples, nrow=1, normalize=False)
        os.makedirs(sample_dir, exist_ok=True)

        img_path = os.path.join(sample_dir, f"iter_{i}_timeline.png")
        save_image(grid, img_path)

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": i + 1,
            "sample":wandb.Image(img_path)
        })


    torch.save(model.state_dict(), checkpoint_path)
    
    wandb.finish()
    
    
if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",type=int, default=200)
    parser.add_argument("--lr",type=float, default=1e-4)
    parser.add_argument("--batch_size",type=int, default=16)
    parser.add_argument("--num_workers",type=int, default=6)
    parser.add_argument("--sigma_min",type=float, default=0)
    parser.add_argument("--inference_step",type=int, default=20)
    parser.add_argument("--model_type",type=str, default="optimal transport")
    
    args = parser.parse_args()
    run(args)
    
    