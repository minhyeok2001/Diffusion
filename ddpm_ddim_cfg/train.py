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

from tqdm import tqdm
from torchvision.utils import make_grid, save_image

def show_prediction(step,valloader,ddpm_scheduler,model,device,out_dir="checkpoints/val_samples",cfg=False,cfg_weight=2.5):
    img, cls = next(iter(valloader))
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
            
            if cfg :
                cond_noise = model(x_t,t,cls)
                uncond_noise = model(x_t,t,torch.zeros_like(cls))
                noise = (1+cfg_weight)*cond_noise - cfg_weight * uncond_noise
            
            else :
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

    img_path = os.path.join(out_dir, f"iter_{step}_timeline.png")
    save_image(grid, img_path)
                            
    return x_t, img_path

def show_prediction_x_0(step,valloader,ddpm_scheduler,model,device,out_dir="checkpoints/val_samples",cfg=False,cfg_weight=2.5):
    img, cls = next(iter(valloader))
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
            t= torch.full((img.shape[0],), t, device=device, dtype=torch.long)

            x_0 = model(x=x_t,t=t,cls=cls)
            _,x_t_1,__ = ddpm_scheduler.reverse_process_x_0(t=t,x_t=x_t,x_0=x_0)
            x_t = x_t_1
        
            t_idx_int = int(t[0].item())
            if t_idx_int in snap_idxs:
                x_t_1 = (x_t_1 +1 )/2
                snapshots.append(x_t_1[:min(8, x_t_1.size(0))])

    samples = torch.cat(snapshots, dim=-1)
    grid = make_grid(samples, nrow=1, normalize=False)
    os.makedirs(out_dir, exist_ok=True)

    img_path = os.path.join(out_dir, f"iter_{step}_timeline.png")
    save_image(grid, img_path)
                            
    return x_t, img_path


def show_prediction_mu(step,valloader,ddpm_scheduler,model,device,out_dir="checkpoints/val_samples",cfg=False,cfg_weight=2.5):
    img, cls = next(iter(valloader))
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
            t= torch.full((img.shape[0],), t, device=device, dtype=torch.long)
    
            mu = model(x=x_t,t=t,cls=cls)
            _,x_t_1,__ = ddpm_scheduler.reverse_process_mu(t=t,x_t=x_t,mu=mu)

            x_t = x_t_1
        
            t_idx_int = int(t[0].item())
            if t_idx_int in snap_idxs:
                x_t_1 = (x_t_1 +1 )/2
                snapshots.append(x_t_1[:min(8, x_t_1.size(0))])

    samples = torch.cat(snapshots, dim=-1)
    grid = make_grid(samples, nrow=1, normalize=False)
    os.makedirs(out_dir, exist_ok=True)

    img_path = os.path.join(out_dir, f"iter_{step}_timeline.png")
    save_image(grid, img_path)
                            
    return x_t, img_path


def show_prediction_x_0_fid(valloader, ddpm_scheduler, model, device, out_dir="checkpoints/val_samples", cfg=True, cfg_weight=2.5):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, "real")
    gen_dir  = os.path.join(out_dir, "gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)

    t_len = len(ddpm_scheduler.timesteps)
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
            for t in range(t_len - 1, -1, -1):
                t= torch.full((img.shape[0],), t, device=device, dtype=torch.long)

                x_0 = model(x=x_t,t=t,cls=cls)
                _,x_t_1,__ = ddpm_scheduler.reverse_process_x_0(t=t,x_t=x_t,x_0=x_0)
                x_t = x_t_1

            gen_imgs = (x_t + 1) / 2 
            for i in range(gen_imgs.size(0)):
                save_image(gen_imgs[i].cpu(), os.path.join(gen_dir, f"{save_idx:06d}.png"))

            save_idx += img.size(0)

    print(f"저장 완료: {real_dir}, {gen_dir}")
    return real_dir, gen_dir


def show_prediction_mu_fid(valloader, ddpm_scheduler, model, device, out_dir="checkpoints/val_samples", cfg=True, cfg_weight=2.5):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, "real")
    gen_dir  = os.path.join(out_dir, "gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)

    t_len = len(ddpm_scheduler.timesteps)
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
            for t in range(t_len - 1, -1, -1):
                t = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
                
                mu = model(x=x_t,t=t,cls=cls)
                _,x_t_1,__ = ddpm_scheduler.reverse_process_mu(t=t,x_t=x_t,mu=mu)
                x_t = x_t_1

            gen_imgs = (x_t + 1) / 2 
            for i in range(gen_imgs.size(0)):
                save_image(gen_imgs[i].cpu(), os.path.join(gen_dir, f"{save_idx:06d}.png"))

            save_idx += img.size(0)

    print(f"저장 완료: {real_dir}, {gen_dir}")
    return real_dir, gen_dir



def run(args):
    
    ## 이렇게 하면 안되지만, colab 이용해야하므로 ..,,
    wandb.login(key="08198b7be027ddffa5241b9acf2f45cd4d42e993")
    
    device = "cuda"
    epoch = args.epoch 
    lr = args.lr 
    batch_size = args.batch_size
    num_workers = args.num_workers
    cfg = args.cfg
    cfg_dropout= args.cfg_dropout
    cfg_weight = args.cfg_weight
    diffusion_type = args.diffusion_type
    ckpt_name = args.ckpt_name
    pred_type = args.pred_type
    
    
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
    ## 2. Model definition & setting stuffs..
    
    model = DiffusionUnet(cfg=cfg,cfg_dropout=cfg_dropout).to(device)
    ddpm_scheduler = DDPMScheduler(inference_step=1000,device=device)
    
    wandb.watch(model, log="all")

    print("model params : ",sum(item.numel() for item in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=epoch)
    
    loss_ft = DiffusionLoss()
    
    #checkpoint_path = "checkpoints/Diffusion_.pth"
    checkpoint_path = os.path.join("checkpoints",ckpt_name)
    
    sample_dir = "checkpoints/val_samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    if pred_type == "eps" or  pred_type == "x_0" or pred_type == "mu":
        pass
    else :
        raise RuntimeError("Networ type Error !!")
    
    ## 3. train loop
    ## method : 배치사이즈만큼의 time step을 랜덤으로 만든다 -> 해당 타임스텝에서의 forward process를 가져온다 -> 그걸 넣고 노이즈를 예측하도록 한다 
    
    ## CFG 사용하기 
    # -> 1. 배치 기준으로 2배로 이미지를 늘리고, 절반은 NULL class, 절반은 1,2,3 class 부여. or 네트워크 내에서는 cfg dropout 비율을 설정하여, 이보다 작은경우는 그냥 NULL 부여 
    # -> 2. inference 시에 eps  = 1+w cond - w uncond 식 사용 

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
            t =torch.randint(0,len(ddpm_scheduler.timesteps),(img.shape[0],), device=device) ## 이게 t idx가 아니라 그냥 t인가?

            ## 2. 해당 t에 맞게 forward process를 한다 with noise_gt
            x_t, noise_gt = ddpm_scheduler.forward_process(t=t,x_0=img)
            
            ## 3. noise 예측 Unet (Option. eps 말고도 mu, x_0 predictor 선택 가능)
            if pred_type == "eps":
                noise_pred = model(x=x_t,t=t,cls=cls)
                loss = loss_ft(noise_pred,noise_gt)    
                
            elif pred_type == "x_0" :
                x_0_pred = model(x=x_t,t=t,cls=cls) ## 이제는 model의 예측은 x_0
                x_0_gt = img
                loss = loss_ft(x_0_pred,x_0_gt)
                
            elif pred_type == "mu" :
                mu_pred = model(x=x_t,t=t,cls=cls) ## 이제는 model의 예측은 mu
                # gt는 그 alpha_cumprod한걸 사용하면 될듯
                mu_gt = ddpm_scheduler.posterior_mean(t=t,x_t=x_t,x_0=img)
                loss = loss_ft(mu_pred,mu_gt)

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
                cls = cls.to(device)
                
                img = img * 2 - 1
                
                t =torch.randint(0,len(ddpm_scheduler.timesteps),(img.shape[0],), device=device)
                
                x_t, noise_gt = ddpm_scheduler.forward_process(t=t,x_0=img)
                
                if pred_type == "eps":
                    noise_pred = model(x=x_t,t=t,cls=cls)
                    loss = loss_ft(noise_pred,noise_gt)    
                    
                elif pred_type == "x_0" :
                    x_0_pred = model(x=x_t,t=t,cls=cls) 
                    x_0_gt = img
                    loss = loss_ft(x_0_pred,x_0_gt)
                    
                elif pred_type == "mu" :
                    mu_pred = model(x=x_t,t=t,cls=cls)
                    mu_gt = ddpm_scheduler.posterior_mean(t=t,x_t=x_t,x_0=img)
                    loss = loss_ft(mu_pred,mu_gt)

                val_loss += loss.item()
            
        avg_val_loss = val_loss / val_batches
        print(f"Epoch [{i+1}/{epoch}] | Val Loss: {avg_val_loss:.6f}")
        #scheduler.step()
        
        if pred_type == "eps" :
            _, img_path = show_prediction(step=i,valloader=visual_valloader,ddpm_scheduler=ddpm_scheduler,model=model,cfg=cfg,device=device,cfg_weight=cfg_weight)
        elif pred_type == "x_0" :
            _, img_path = show_prediction_x_0(step=i,valloader=visual_valloader,ddpm_scheduler=ddpm_scheduler,model=model,cfg=cfg,device=device,cfg_weight=cfg_weight)
        elif pred_type == "mu" :
            _, img_path = show_prediction_mu(step=i,valloader=visual_valloader,ddpm_scheduler=ddpm_scheduler,model=model,cfg=cfg,device=device,cfg_weight=cfg_weight)

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": i + 1,
            "sample":wandb.Image(img_path)
        })


    torch.save(model.state_dict(), checkpoint_path)
    
    
    wandb.finish()
    
    ## 바로 fid score 출력하도록,..
    
    if pred_type == "x_0" :
        show_prediction_x_0_fid(valloader=visual_valloader,ddpm_scheduler=ddpm_scheduler,model=model,device=device)
    elif pred_type == "mu" :
        show_prediction_mu_fid(valloader=visual_valloader,ddpm_scheduler=ddpm_scheduler,model=model,device=device)
             
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.2)
    parser.add_argument("--cfg_weight", type=float, default=2.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--diffusion_type", type=str, default="ddpm",choices=["ddpm", "ddim"],help="Choose which diffusion algorithm to use: 'ddpm' or 'ddim'.")
    parser.add_argument("--ckpt_name",type=str)
    parser.add_argument("--pred_type",type=str)
    
    args = parser.parse_args()
    
    run(args)
