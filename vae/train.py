from .vae_model import *
import data.get_data 
import data.dataloader 


import torch
import torch.nn as nn
import argparse


def run(args):
    
    epoch = args.epoch 
    lr = args.lr 
    batch_size = args.batch_size

    ## wandb는 우선 패스. 
    
    ## 1. Data preparation
    
    dataset = data.dataloader.CustomDataset()
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size)
    
    
    
    ## 2. Model definition

    model = VAE([64,128,256])
    print(sum(item.numel() for item in model.parameters()))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=3) # 한개가 3개의 이미지셋을 다루므로.. 배치사이즈 3만 해도 사진 9장
    
    args = parser.parse_args()
    
    run(args)