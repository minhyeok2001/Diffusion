import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionLoss(nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self,pred,gt):
        return F.mse_loss(pred,gt)