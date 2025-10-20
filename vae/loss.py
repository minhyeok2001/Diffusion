import torch.nn as nn
import torch
import torch.nn.functional as F

class VaeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def reconstruction_term(self,pred,gt):
        return F.mse_loss(pred,gt)
        
    def matching_term(self,mu,sigma):
        kl = -1/2 * ( 1 + torch.log(sigma**2) - sigma**2 - mu**2 )
        return kl.mean()
    
    def forward(self,pred,gt,mu,sigma):
        loss = self.matching_term(mu,sigma)
        loss += self.reconstruction_term(pred,gt)
        return loss