import data.get_data 
import data.dataloader 
from .vae_model import *
from .loss import *

import os
import torch
import torch.nn as nn
import argparse

from tqdm import tqdm
from torchvision.utils import make_grid, save_image



def show_prediction(valloader, model, device="cuda", sample_dir="checkpoints/val_samples"):
    model.eval()
    os.makedirs(sample_dir, exist_ok=True)

    real_dir = os.path.join(sample_dir,"real")
    gen_dir  = os.path.join(sample_dir,"gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)

    save_idx = 0

    with torch.no_grad():
        for img, cls in tqdm(valloader):

            img = img.to(device)
            pred, mu, sigma = model(img) 

            originals = img.detach().cpu()
            recon = pred.detach().cpu()

            bsz = originals.size(0)
            for i in range(bsz):
                save_image(originals[i], os.path.join(real_dir, f"{save_idx:06d}.png"))
                save_image(recon[i],     os.path.join(gen_dir,  f"{save_idx:06d}.png"))
                save_idx += 1
                
                
def run(args):
    device = "cuda"
    epoch = args.epoch 
    lr = args.lr 
    batch_size = args.batch_size
    num_workers = args.num_workers

    ## wandb는 우선 패스. 
    
    ## 1. Data preparation
    
    dataset = data.dataloader.CustomDataset()
    trainloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,collate_fn=data.dataloader.collate_ft,num_workers= num_workers)
    
    valset = data.dataloader.CustomDataset(test=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=batch_size,num_workers= num_workers)
    
    ## 2. Model definition & setting stuffs..

    model = VAE([128,256,512]).to(device)
    print("model params : ",sum(item.numel() for item in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=epoch)
    
    loss_ft = VaeLoss()
    beta = 0.3

    checkpoint_path = "checkpoints/VAE.pth"
    
    sample_dir = "checkpoints/val_samples"
    os.makedirs(sample_dir, exist_ok=True)


    ## 3. train loop
    for i in range(epoch) :
        running_loss = 0.0
        total_len = len(trainloader)
        for img, cls in tqdm(trainloader) :
            model.train()
            optimizer.zero_grad()
            
            img = img.to(device)
            cls = cls.to(device)
            
            ## VAE 만을 위한 전처리이므로, 데이터로더에서 처리하지말고 여기서 처리. 데이터로더에서는 디퓨전 구현시에도 사용해야하므로..
            #img = img*2-1 -> 그냥 출력을 sigmoid로 감싸는 것으로 구현
            
            pred, mu, sigma= model(img)
            mt,rc = loss_ft(pred,img, mu, sigma)
            
            loss =  beta * mt + rc
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            #print("matching term per batch :",  mt.item())
            #print("reconstruction term per batch :", rc.item())
            #print("total loss :",loss.item())
 
        avg_train_loss = running_loss / total_len
        print(f"Epoch [{i+1}/{epoch}] | Train Loss: {avg_train_loss:.6f}")
    
        
        val_loss = 0.0
        val_batches = len(valloader)
        with torch.no_grad():
            for idx,(img, cls) in tqdm(enumerate(valloader)):
                model.eval()
                img = img.to(device)
                cls = cls.to(device)

                #img = img*2-1
                
                pred, mu, sigma= model(img)
                mt,rc = loss_ft(pred,img, mu, sigma)
                
                loss =  beta * mt + rc
                    
                val_loss += loss.item()
                
                if idx == 0 :
                    num_show = min(4, img.size(0))
                    originals = img[:num_show].cpu()
                    pred, mu, sigma = model(img)
                    recon = pred[:num_show].cpu()
                    stacked = torch.stack([originals, recon], dim=1).flatten(0, 1)
                    grid = make_grid(stacked, nrow=num_show, normalize=False, value_range=(0, 1))
                    save_image(grid, os.path.join(sample_dir, f"reconstructed_img_epoch_{i+1}.png"))
            
        avg_val_loss = val_loss / val_batches
        print(f"Epoch [{i+1}/{epoch}] | Val Loss: {avg_val_loss:.6f}")

        scheduler.step()
        
    show_prediction(valloader=valloader,model=model)


    torch.save(model.state_dict(), checkpoint_path)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1) # 한개가 3개의 이미지셋을 다루므로.. 배치사이즈 3만 해도 사진 9장
    
    args = parser.parse_args()
    
    run(args)
