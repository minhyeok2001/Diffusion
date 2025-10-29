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
from diffusers import AutoencoderKL


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
                

class HFVAEAdapter(nn.Module):
    def __init__(self, repo_id="stabilityai/sd-vae-ft-mse", train_vae=True):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(repo_id)
        if not train_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _to_minus1_1(self, x):
        return x * 2.0 - 1.0

    @torch.no_grad()
    def _to_0_1(self, x):
        return (x + 1.0) / 2.0

    def forward(self, img):
        x_in = self._to_minus1_1(img)

        enc_out = self.vae.encode(x_in)
        posterior = enc_out.latent_dist               
        mu = posterior.mean

        sigma = torch.exp(0.5 * posterior.logvar)

        z = posterior.sample()

        dec = self.vae.decode(z) 
        x_rec_minus1_1 = dec.sample
        recon = self._to_0_1(x_rec_minus1_1)

        return recon, mu, sigma
                
def run(args):
    device = "cuda"
    epoch = args.epoch 
    lr = args.lr 
    batch_size = args.batch_size
    num_workers = args.num_workers
    beta = args.beta

    ## wandb는 우선 패스. 
    
    valset = data.dataloader.CustomDataset(test=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=batch_size,num_workers= num_workers)
    
    ## 2. Model definition & setting stuffs..

    model = HFVAEAdapter("stabilityai/sd-vae-ft-mse", train_vae=False).to(device)
    print("model params : ",sum(item.numel() for item in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=epoch)
    
    loss_ft = VaeLoss()

    checkpoint_path = "checkpoints/VAE.pth"
    
    sample_dir = "checkpoints/val_samples"
    os.makedirs(sample_dir, exist_ok=True)


    ## 3. train loop
    for i in range(1) :
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


    #torch.save(model.state_dict(), checkpoint_path)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--beta", type=float, default= 0.1)
    parser.add_argument("--batch_size", type=int, default=1) 
    
    args = parser.parse_args()
    
    run(args)
